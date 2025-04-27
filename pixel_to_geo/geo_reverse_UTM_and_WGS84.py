import json
import subprocess
import numpy as np
import cv2
from pyproj import Transformer
import exifread
import piexif
from fractions import Fraction
import re 
import rasterio

def sample_elevation(dtm_path, x_utm, y_utm):
    """
    Returns ground elevation (in meters) from dtm.tif at given UTM coords.
    """
    with rasterio.open(dtm_path) as src:
        # Transform world coords to pixel row/col
        row, col = src.index(x_utm, y_utm)
        # Read the single-band elevation
        elev = src.read(1)[row, col]
    return float(elev)


# 1. Load camera intrinsics from camera.json
def load_intrinsics(camera_json="camera.json"):
    cam = json.load(open(camera_json))
    # Assuming only one key; take its parameters
    params = next(iter(cam.values()))
    K = np.array([
        [params["focal_x"],             0, params["c_x"]],
        [            0, params["focal_y"], params["c_y"]],
        [            0,             0,          1   ]
    ], dtype=float)
    dist = np.array([
        params.get("k1",0), params.get("k2",0),
        params.get("p1",0), params.get("p2",0),
        params.get("k3",0)
    ], dtype=float)
    return K, dist

# 2. Extract GPS & orientation (roll, pitch, yaw) via exiftool
def dms_to_dd(dms, ref):
    # exifread returns DMS as e.g. [49, 5, 57.8848]
    deg, min, sec = [float(x.num) / float(x.den) for x in dms.values]
    dd = deg + min/60 + sec/3600
    return -dd if ref in ("S","W") else dd

def load_exif_pose(image_path):
    # --- 1) Read EXIF GPS tags via exifread ---
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    # Latitude / Longitude
    lat = dms_to_dd(tags["GPS GPSLatitude"],  tags["GPS GPSLatitudeRef"].printable)
    lon = dms_to_dd(tags["GPS GPSLongitude"], tags["GPS GPSLongitudeRef"].printable)
    # Altitude (meters)
    alt = float(tags.get("GPS GPSAltitude", 0).values[0].num) / float(tags.get("GPS GPSAltitude", 0).values[0].den)

    # --- 2) Read XMP for DJI flight angles via regex ---
    # (fast and light; assumes the file’s <x:xmpmeta> is UTF-8)
    with open(image_path, "rb") as f:
        raw = f.read().decode("utf-8", "ignore")

    def find_angle(tag):
        m = re.search(rf"<drone-dji:{tag}>([-\d\.]+)</drone-dji:{tag}>", raw)
        return float(m.group(1)) if m else 0.0

    roll  = find_angle("FlightRollDegree")
    pitch = find_angle("FlightPitchDegree")
    yaw   = find_angle("FlightYawDegree")

    return lat, lon, alt, roll, pitch, yaw

# 3. Build extrinsic R, t from UTM + orientation
def build_extrinsics(lat, lon, alt, roll, pitch, yaw, zone, offset_e, offset_n):
    # Convert lat/lon → UTM (meters)
    # Always use (lon, lat) order with always_xy=True
    transformer = Transformer.from_crs("EPSG:4326",
                                       f"+proj=utm +zone={zone[:-1]} +{zone[-1].lower()} +datum=WGS84 +units=m +no_defs",
                                       always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    # Local camera center in UTM: subtract integer offset
    C = np.array([easting - offset_e,
                  northing - offset_n,
                  alt]).reshape(3,1)
    # Build rotation from Roll, Pitch, Yaw (in degrees → radians)
    r, p, y = np.deg2rad([roll, pitch, yaw])
    # ZYX (yaw→pitch→roll) intrinsic rotations:
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
    # Compute translation: t = -R·C
    t = -R @ C
    return R, t

# 4. Read UTM offset (coords.txt)
def load_utm_offset(path="odm_georeferencing/coords.txt"):
    lines = [l.strip() for l in open(path) if l.strip()]
    zone = lines[0].split()[-1]  # e.g. "32N"
    off_e, off_n = map(float, lines[1].split()[:2])
    return off_e, off_n, zone

# 5. Undistort & back-project to local ground plane z=0
def pixel_to_local_xyz(u, v, K, dist, R, t):
    # normalize & undistort
    pt = np.array([[[u, v]]], dtype=float)
    n = cv2.undistortPoints(pt, K, dist, P=np.eye(3))[0][0]
    ray_cam = np.array([n[0], n[1], 1.0]).reshape(3,1)
    # camera center in world
    C = -R.T @ t
    dir_w = R.T @ ray_cam
    # intersect z=0: C_z + λ·d_z = 0 → λ = -C_z/d_z
    lam = -C[2,0] / dir_w[2,0]
    X = C + lam * dir_w
    return X.flatten()  # (x_local, y_local, 0)

# 6. Convert UTM→lat/lon for final result
def utm_to_latlon(x, y, zone):
    proj_str = f"+proj=utm +zone={zone[:-1]} +{zone[-1].lower()} +datum=WGS84 +units=m +no_defs"
    print(proj_str)
    # proj_str = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +type=crs"
    transformer = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=False)
    lon, lat = transformer.transform(x, y)
    return lat, lon

# 7. Main API
def pixel_to_latlon(image_path, u, v,
                    camera_json="camera.json",
                    coords_txt="odm_georeferencing/coords.txt",
                    dtm_path="odm_dem/dtm.tif"):
    # … load K, dist, UTM offset, and extrinsics as before …
    K, dist = load_intrinsics(camera_json)
    # parse UTM offset
    off_e, off_n, zone = load_utm_offset(coords_txt)
    # parse EXIF pose
    lat, lon, alt, roll, pitch, yaw = load_exif_pose(image_path)
    # build extrinsics
    R, t = build_extrinsics(lat, lon, alt, roll, pitch, yaw,
                            zone, off_e, off_n)
    
        # Undistort & get normalized coords
    pt = np.array([[[u, v]]], dtype=np.float32)
    n  = cv2.undistortPoints(pt, K, dist, P=np.eye(3))[0][0]
    norm_x, norm_y = n[0], n[1]

    # Now build the world‐ray
    ray_cam = np.array([norm_x, norm_y, 1.0]).reshape(3,1)
    dir_w   = R.T @ ray_cam

    # Back-project pixel into local frame
    x_l, y_l, _ = pixel_to_local_xyz(u, v, K, dist, R, t)
    # Global UTM coords
    x_utm, y_utm = off_e + x_l, off_n + y_l

    # 1) Sample true ground elevation
    z_ground = sample_elevation(dtm_path, x_utm, y_utm)

    # 2) Recompute 3D intersection with terrain
    C = -R.T @ t
    dir_w = R.T @ np.array([norm_x, norm_y, 1.0]).reshape(3,1)
    lam = (z_ground - C[2,0]) / dir_w[2,0]
    X = C + lam * dir_w

    # 3) Convert UTM→lat/lon as before
    lat_pt, lon_pt = utm_to_latlon(X[0,0] + off_e, X[1,0] + off_n, zone)
    return lat_pt, lon_pt, z_ground

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Map pixel to lat/lon using camera.json & EXIF"
    )
    p.add_argument("image", help="Image file path")
    p.add_argument("u",    type=float, help="Pixel X")
    p.add_argument("v",    type=float, help="Pixel Y")
    args = p.parse_args()
    # lat, lon = pixel_to_latlon(args.image, args.u, args.v, camera_json=r"nodeodm_out_4\cameras.json", coords_txt=r"nodeodm_out_4\odm_georeferencing\coords.txt")
    # print(f"{args.image} ({args.u:.1f},{args.v:.1f}) →  lat: {lat:.8f}, lon: {lon:.8f}")
    lat, lon, alt = pixel_to_latlon(
        args.image, args.u, args.v,
        camera_json=r"nodeodm_test_1\cameras.json",
        coords_txt=r"nodeodm_test_1\odm_georeferencing\coords.txt",
        dtm_path=r"nodeodm_test_1\odm_dem\dem.tif"
    )
    print(
        f"{args.image} ({args.u:.1f},{args.v:.1f}) →  "
        f"lat: {lat:.8f}, lon: {lon:.8f}, alt: {alt:.3f} m"
    )
