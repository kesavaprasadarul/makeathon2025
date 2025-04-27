from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
import rasterio
import cv2
from pyproj import Transformer
import exifread
import re
import shutil
import tempfile
import os

app = FastAPI()

# ---------- Your existing helper functions ----------

class DataSource:
    def __init__(self, name, folderpath, images_folder_path):
        self.name = name
        self.set_folderpath(folderpath)
        self.set_images_folder_path(images_folder_path)

    def set_folderpath(self, folderpath):
        self.folderpath = folderpath
        self.camera_json_path = os.path.join(folderpath, "cameras.json")
        self.coords_path = os.path.join(folderpath, "odm_georeferencing/coords.txt")
        self.dtm_path = os.path.join(folderpath, "odm_dem/dtm.tif")
        self._dtm_src = rasterio.open(self.dtm_path)
        self._dtm_arr = self._dtm_src.read(1)
        self._dtm_bounds = self._dtm_src.bounds  # (minx, miny, maxx, maxy)
    
    def set_images_folder_path(self, images_folder_path):
        self.images_folder_path = images_folder_path
        self.image_paths = [os.path.join(images_folder_path, f) for f in os.listdir(images_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]


    def sample_elevation(self, x_utm: float, y_utm: float, clamp: bool = True) -> float:
        minx, miny, maxx, maxy = self._dtm_bounds
        # clamp or error if outside
        if not (minx <= x_utm <= maxx and miny <= y_utm <= maxy):
            if clamp:
                x_utm = min(max(x_utm, minx), maxx)
                y_utm = min(max(y_utm, miny), maxy)
            else:
                raise HTTPException(400, f"Point ({x_utm:.2f}, {y_utm:.2f}) outside DTM bounds")
        row, col = self._dtm_src.index(x_utm, y_utm)
        return float(self._dtm_arr[row, col])
    
regensburg = DataSource(name ="regensburg", 
                    folderpath = "/datasets/nodeodm_out_4",
                    images_folder_path ="/datasets/images/regensburg")

munich = DataSource(name ="munich",
                    folderpath = "/datasets/nodeodm_test_1",
                    images_folder_path ="/datasets/images/munich")


def load_intrinsics(camera_json_path):
    import json
    cam = json.load(open(camera_json_path))
    params = next(iter(cam.values()))
    K = np.array([
        [params["focal_x"], 0, params["c_x"]],
        [0, params["focal_y"], params["c_y"]],
        [0, 0, 1]
    ], dtype=float)
    dist = np.array([
        params.get("k1",0), params.get("k2",0),
        params.get("p1",0), params.get("p2",0),
        params.get("k3",0)
    ], dtype=float)
    return K, dist

def dms_to_dd(dms, ref):
    deg, min, sec = [float(x.num) / float(x.den) for x in dms.values]
    dd = deg + min/60 + sec/3600
    return -dd if ref in ("S","W") else dd

def load_exif_pose(image_path):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    lat = dms_to_dd(tags["GPS GPSLatitude"],  tags["GPS GPSLatitudeRef"].printable)
    lon = dms_to_dd(tags["GPS GPSLongitude"], tags["GPS GPSLongitudeRef"].printable)
    alt = float(tags.get("GPS GPSAltitude", 0).values[0].num) / float(tags.get("GPS GPSAltitude", 0).values[0].den)
    with open(image_path, "rb") as f:
        raw = f.read().decode("utf-8", "ignore")
    def find_angle(tag):
        m = re.search(rf"<drone-dji:{tag}>([-\d\.]+)</drone-dji:{tag}>", raw)
        return float(m.group(1)) if m else 0.0
    roll  = find_angle("FlightRollDegree")
    pitch = find_angle("FlightPitchDegree")
    yaw   = find_angle("FlightYawDegree")
    return lat, lon, alt, roll, pitch, yaw

def build_extrinsics(lat, lon, alt, roll, pitch, yaw, zone, offset_e, offset_n):
    transformer = Transformer.from_crs("EPSG:4326",
                                       f"+proj=utm +zone={zone[:-1]} +{zone[-1].lower()} +datum=WGS84 +units=m +no_defs",
                                       always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    C = np.array([easting - offset_e,
                  northing - offset_n,
                  alt]).reshape(3,1)
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                   [ np.sin(y),  np.cos(y), 0],
                   [        0 ,         0 , 1]])
    Ry = np.array([[  np.cos(p), 0, np.sin(p)],
                   [          0 , 1,        0],
                   [ -np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1,         0 ,          0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    R = Rz @ Ry @ Rx
    t = -R @ C
    return R, t

def load_utm_offset(coords_path):
    lines = [l.strip() for l in open(coords_path) if l.strip()]
    zone = lines[0].split()[-1]
    off_e, off_n = map(float, lines[1].split()[:2])
    return off_e, off_n, zone

def pixel_to_local_xyz(u, v, K, dist, R, t):
    pt = np.array([[[u, v]]], dtype=float)
    n = cv2.undistortPoints(pt, K, dist, P=np.eye(3))[0][0]
    ray_cam = np.array([n[0], n[1], 1.0]).reshape(3,1)
    C = -R.T @ t
    dir_w = R.T @ ray_cam
    lam = -C[2,0] / dir_w[2,0]
    X = C + lam * dir_w
    return X.flatten()

def utm_to_latlon(x, y, zone):
    proj_str = f"+proj=utm +zone={zone[:-1]} +{zone[-1].lower()} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=False)
    lon, lat = transformer.transform(x, y)
    return lat, lon

def pixel_to_geo_image(datasource: DataSource, image_file_name: str, x: float, y: float):
    # Load the image and camera parameters
    K, dist = load_intrinsics(datasource.camera_json_path)
    
    # Load the EXIF data to get the pose
    lat, lon, alt, roll, pitch, yaw = load_exif_pose(image_file_name)
    
    # Load the UTM offset from coords.txt
    off_e, off_n, zone = load_utm_offset(datasource.coords_path)
    
    # Build the extrinsics
    R, t = build_extrinsics(lat, lon, alt, roll, pitch, yaw, zone, off_e, off_n)

    pt = np.array([[[x, y]]], dtype=np.float32)
    n = cv2.undistortPoints(pt, K, dist, P=np.eye(3))[0][0]
    norm_x, norm_y = n[0], n[1]

    C = -R.T @ t
    dir_w = R.T @ np.array([norm_x, norm_y, 1.0]).reshape(3,1)

    x_l, y_l, _ = pixel_to_local_xyz(x, y, K, dist, R, t)
    x_utm, y_utm = off_e + x_l, off_n + y_l
    z_ground = datasource.sample_elevation(x_utm, y_utm)

    lam = (z_ground - C[2,0]) / dir_w[2,0]
    X = C + lam * dir_w

    lat_pt, lon_pt = utm_to_latlon(X[0,0] + off_e, X[1,0] + off_n, zone)
    return lat_pt, lon_pt, z_ground

def pixel_to_geo_image_fast(x: float, y: float, K, dist, R, t, off_e, off_n, zone, datasource: DataSource):
    pt = np.array([[[x, y]]], dtype=np.float32)
    n = cv2.undistortPoints(pt, K, dist, P=np.eye(3))[0][0]
    norm_x, norm_y = n[0], n[1]

    C = -R.T @ t
    dir_w = R.T @ np.array([norm_x, norm_y, 1.0]).reshape(3,1)

    x_l, y_l, _ = pixel_to_local_xyz(x, y, K, dist, R, t)
    x_utm, y_utm = off_e + x_l, off_n + y_l
    z_ground = datasource.sample_elevation(x_utm, y_utm)

    lam = (z_ground - C[2,0]) / dir_w[2,0]
    X = C + lam * dir_w

    lat_pt, lon_pt = utm_to_latlon(X[0,0] + off_e, X[1,0] + off_n, zone)
    return lat_pt, lon_pt, z_ground

# ---------- FastAPI Endpoint ----------

@app.post("/pixel-to-latlon")
async def pixel_to_latlon_api(
    imageName: str = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    datasource_name: str = Form(...)
):
    # Select the data source based on the name provided
    if datasource_name == "regensburg":
        datasource = regensburg
    elif datasource_name == "munich":
        datasource = munich
    else:
        return JSONResponse(content={"error": "Invalid data source name"}, status_code=400)

    # Construct the full image path
    image_path = os.path.join(datasource.images_folder_path, imageName)

    # Check if the image file exists
    if not os.path.exists(image_path):
        return JSONResponse(content={"error": "Image file not found"}, status_code=404)

    # Process the image to get latitude, longitude, and altitude
    lat_pt, lon_pt, z_ground = pixel_to_geo_image(
        datasource,
        image_path,
        x,
        y
    )
    
    return JSONResponse(content={
        "latitude": lat_pt,
        "longitude": lon_pt,
        "altitude": z_ground
    })
    
class PixelToLatLonRequestMulti(BaseModel):
    imageName: List[str]
    u: List[float]
    v: List[float]
    datasource_name: str
    
class PixelToLatLonRequest(BaseModel):
    imageName: str
    u: List[float]
    v: List[float]
    datasource_name: str
    
    
@app.post("/pixel-to-latlon/batch")
async def pixel_to_latlon_one_image_many_pixels(request: PixelToLatLonRequestMulti):
    # request.imageName is a List[str], but we enforce it has exactly one element:
    if len(request.imageName) != 1:
        raise HTTPException(400, "This endpoint only accepts exactly one imageName")
    img = request.imageName[0].strip()

    # pick data source
    if request.datasource_name == "regensburg":
        datasource = regensburg
    elif request.datasource_name == "munich":
        datasource = munich
    else:
        raise HTTPException(400, "Invalid data source name")

    image_path = os.path.join(datasource.images_folder_path, img)
    if not os.path.exists(image_path):
        raise HTTPException(404, f"Image {img} not found")

    # preload everything once
    K, dist = load_intrinsics(datasource.camera_json_path)
    lat0, lon0, alt0, roll, pitch, yaw = load_exif_pose(image_path)
    off_e, off_n, zone = load_utm_offset(datasource.coords_path)
    R, t = build_extrinsics(lat0, lon0, alt0, roll, pitch, yaw, zone, off_e, off_n)

    results = []
    # now loop *only* over your pixels
    for u_val, v_val in zip(request.u, request.v):
        lat_pt, lon_pt, z_ground = pixel_to_geo_image_fast(
            u_val, v_val,
            K, dist, R, t,
            off_e, off_n, zone,
            datasource
        )
        results.append({
            "image": img,
            "latitude": lat_pt,
            "longitude": lon_pt,
            "altitude": z_ground
        })

    return JSONResponse(content=results)