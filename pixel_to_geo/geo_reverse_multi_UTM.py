import json
import rasterio
import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from scipy.spatial.transform import Rotation
import os 
# Configuration
IMAGE_FILE = r"C:\Repos\makeathon2025\dev_data\DJI_20250424193303_0155_V.jpeg"
PIXEL_COORDS = (100, 1600)  # (u, v) pixel coordinates to convert
CAMERA_MODEL_KEY = "dji m4td 4032 3024 brown 0.6666"
CAMERA_JSON = r"C:\Repos\makeathon2025\Kesava\nodeodm_out_4\cameras.json"
IMAGES_JSON = r"C:\Repos\makeathon2025\Kesava\nodeodm_out_4\images.json"
COORDS_TXT = r"C:\Repos\makeathon2025\Kesava\nodeodm_out_4\odm_georeferencing\coords.txt"
DSM_TIF = r"C:\Repos\makeathon2025\Kesava\nodeodm_out_4\odm_dem\dsm_UTM32.tif"
# Load camera parameters
with open(CAMERA_JSON) as f:
    cameras = json.load(f)
camera = cameras[CAMERA_MODEL_KEY]

# Calculate ACTUAL focal lengths in pixels
focal_x_pixels = camera['focal_x'] * camera['width']
focal_y_pixels = camera['focal_y'] * camera['height']

# Load images metadata
with open(IMAGES_JSON) as f:
    images = json.load(f)
    
# Find image metadata
image_meta = next(img for img in images if img["filename"] == os.path.basename(IMAGE_FILE))

# Coordinate transformation setup
utm_crs = CRS.from_proj4("+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs")
utm32_crs = CRS.from_epsg(32632)  # Your project CRS (from proj.txt)
wgs84_crs = CRS.from_epsg(4326)
transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

# Convert image GPS to UTM
# FIX 1: Use correct coordinate order (lon, lat) for UTM conversion
transformer_wgs84_to_utm = Transformer.from_crs(wgs84_crs, utm_crs)
transformer_utm_to_wgs84 = Transformer.from_crs(utm_crs, wgs84_crs)
transformer_wgs84_to_utm32 = Transformer.from_crs(wgs84_crs, utm32_crs)
transformer_utm32_to_wgs84 = Transformer.from_crs(utm32_crs, wgs84_crs)

# Convert image GPS to UTM (FIX: longitude first!)
utm32_x, utm32_y = transformer_wgs84_to_utm32.transform(
    image_meta["longitude"],  # X = longitude
    image_meta["latitude"]    # Y = latitude
)

utm33_x= utm32_x
utm33_y= utm32_y
# utm33_x, utm33_y = transformer_utm32_to_utm33.transform(utm32_x, utm32_y)

# FIX 2: Verify altitude reference
with rasterio.open(DSM_TIF) as dsm:
    assert dsm.crs == utm32_crs
    camera_elevation = list(dsm.sample([(utm33_x, utm33_y)]))[0][0]
    height_above_ground = image_meta["altitude"] - camera_elevation


# Enhanced undistortion function
def undistort_pixel(u, v):
    """Brown's distortion model with proper focal length handling"""
    # Principal point coordinates
    cx = camera['width']/2 + camera['c_x']
    cy = camera['height']/2 + camera['c_y']
    
    # Normalized coordinates
    x = (u - cx) / focal_x_pixels
    y = (v - cy) / focal_y_pixels

    # Iterative undistortion
    x_undist, y_undist = x, y
    for _ in range(5):  # Increased iterations for convergence
        r2 = x_undist**2 + y_undist**2
        radial = 1 + camera['k1']*r2 + camera['k2']*(r2**2) + camera['k3']*(r2**3)
        x_dist = x_undist * radial + 2*camera['p1']*x_undist*y_undist + camera['p2']*(r2 + 2*x_undist**2)
        y_dist = y_undist * radial + camera['p1']*(r2 + 2*y_undist**2) + 2*camera['p2']*x_undist*y_undist
        
        # Update estimates
        x_undist = x - x_dist
        y_undist = y - y_dist

    # Convert back to pixel coordinates
    u_undist = x_undist * focal_x_pixels + cx
    v_undist = y_undist * focal_y_pixels + cy
    
    return u_undist, v_undist


# Undistort pixel coordinates
u_undist, v_undist = undistort_pixel(*PIXEL_COORDS)
print(f"Undistorted pixel: {u_undist:.1f}, {v_undist:.1f} (Original: {PIXEL_COORDS})")

# Convert to normalized camera coordinates
x_normalized = (u_undist - (camera['width']/2 + camera['c_x'])) / focal_x_pixels
y_normalized = (v_undist - (camera['height']/2 + camera['c_y'])) / focal_y_pixels
z_normalized = 1.0


# Create rotation matrix from omega, phi, kappa (X, Y, Z rotations)
rotation = Rotation.from_euler(
    'zyx',  # Common drone rotation order (yaw, pitch, roll)
    [
        np.radians(image_meta["kappa"]),   # Z rotation (yaw)
        np.radians(image_meta["phi"]),     # Y rotation (pitch)
        np.radians(image_meta["omega"])    # X rotation (roll)
    ],
    degrees=False
)
R = rotation.as_matrix()

# Calculate ray direction
ray_dir = np.array([x_normalized, y_normalized, z_normalized])
ray_dir /= np.linalg.norm(ray_dir)  # Normalize
rotated_ray = R @ ray_dir

# FIX 6: Iterative ground intersection with DSM
with rasterio.open(DSM_TIF) as dsm:
    transform = dsm.transform
    inv_transform = ~transform
    
    # Start from camera position in UTM33
    x_utm33, y_utm33 = utm33_x, utm33_y  # Start in DSM's CRS
    z_utm = image_meta["altitude"]
    print(inv_transform)
    for _ in range(5):
        # Convert to DSM pixel coordinates
        col, row = inv_transform * (x_utm33, y_utm33)
        print(col, row, x_utm33, y_utm33)
        try:
            print(dsm.bounds)
            ground_z = dsm.read(1)[int(col), int(row)]
        except IndexError:
            print(f"Error: Point ({x_utm33:.2f}, {y_utm33:.2f}) → DSM pixel ({col}, {row}) out of bounds")
            break
            
        # Calculate height above ground
        height = z_utm - ground_z
        
        # Move along ray IN UTM33 COORDINATES
        step = height / rotated_ray[2]
        x_utm33 += rotated_ray[0] * step  # Update previous position
        y_utm33 += rotated_ray[1] * step
        z_utm = ground_z + height_above_ground
        print(f"Iteration {_}: UTM33 ({x_utm33:.2f}, {y_utm33:.2f}) → DSM Elevation: {ground_z:.2f}m")

        
lon, lat = transformer_utm32_to_wgs84.transform(x_utm33, y_utm33)

print(f"Image GPS: {image_meta['latitude']:.6f}, {image_meta['longitude']:.6f}")
print(f"Calculated GPS: {lat:.6f}, {lon:.6f}")
print(f"Offset: {111111 * abs(lat - image_meta['latitude']):.1f} meters")