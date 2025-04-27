#!/usr/bin/env python3
"""
pixel2geo.py: Convert pixel coordinates in a georeferenced orthophoto GeoTIFF
to geographic coordinates (longitude, latitude).
"""

import argparse
import rasterio
from rasterio.transform import xy
from pyproj import Transformer

def pixel_to_geo(ortho_path: str, col: int, row: int):
    """
    Convert a pixel (col, row) in a GeoTIFF orthophoto to (lon, lat).
    """
    with rasterio.open(ortho_path) as src:
        # Read the affine transform and CRS from the dataset
        transform = src.transform
        crs = src.crs

        # Compute projected coordinates (X_geo, Y_geo) at pixel center
        x_geo, y_geo = xy(transform, row, col, offset='center')

        # If already geographic (lat/lon), skip reprojection
        if crs.is_geographic:
            lon, lat = x_geo, y_geo
        else:
            # Reproject from source CRS to WGS84
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x_geo, y_geo)

    return lon, lat

def main():
    parser = argparse.ArgumentParser(
        description="Convert pixel coords in a GeoTIFF orthophoto to lon/lat"
    )
    parser.add_argument("ortho_path", help="Path to the GeoTIFF orthophoto")
    parser.add_argument("col", type=int, help="Pixel column (x index)")
    parser.add_argument("row", type=int, help="Pixel row (y index)")
    args = parser.parse_args()

    lon, lat = pixel_to_geo(args.ortho_path, args.col, args.row)
    print(f"Pixel ({args.col}, {args.row}) → Longitude: {lon:.6f}, Latitude: {lat:.6f}")

if __name__ == "__main__":
    main()
