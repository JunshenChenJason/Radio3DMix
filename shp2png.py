import matplotlib.pyplot as plt
import os
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
import numpy as np
from PIL import Image
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "Beijing_Dataset")
shp_path = os.path.join(DEFAULT_DATA_DIR, "256_square")
output_dir = os.path.join(DEFAULT_DATA_DIR, "BJ_256_png")
output_dir_dilation = os.path.join(DEFAULT_DATA_DIR, "BJ_256_png_dilation")


def shp_to_pixel(shp_path, output_dir, shape=(256, 256), max_height=None):
    os.makedirs(output_dir, exist_ok=True)
    for shp_file in os.listdir(shp_path):
        if shp_file.endswith('.shp'):
            # Load shapefile
            gdf = gpd.read_file(os.path.join(shp_path, shp_file))

            # Make sure that the height list is numerical
            gdf['height'] = gdf['height'].apply(lambda x: x if not math.isnan(x) else 0)

            # Obtain the actual maximum height
            actual_max_height = int(gdf['height'].max()) if not gdf.empty else 0

            # Determine the maximum height to be generated
            use_max_height = max_height if max_height is not None else actual_max_height

            # Get geographic bounds
            xmin, ymin, xmax, ymax = gdf.total_bounds if not gdf.empty else (0, 0, 1, 1)

            # Create affine transformation matrix
            transform = from_bounds(xmin, ymin, xmax, ymax, shape[1], shape[0])

            # Generate the height layer from 1 to use_max_height
            for height_level in range(1, use_max_height + 1):
                # Create an empty grid if the current height layer is greater than the actual maximum height
                if height_level > actual_max_height:
                    # Create a background image
                    img_data = np.full((shape[0], shape[1]), 0, dtype=np.uint8)
                else:
                    # Create a height filter
                    height_filter = (gdf['height'] >= height_level)

                    # Rasterize
                    raster = rasterio.features.rasterize(
                        [(geom, 1) for geom in gdf[height_filter].geometry],
                        out_shape=shape,
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )

                    img_data = np.zeros((shape[0], shape[1]), dtype=np.uint8)
                    img_data[raster == 1] = 255  # 白色填充

                # 保存为PNG，文件名包含高度信息
                output_filename = f"{os.path.splitext(shp_file)[0]}_{height_level}.png"
                Image.fromarray(img_data).save(
                    os.path.join(output_dir, output_filename)
                )


def shp_to_pixel_dilation(shp_path, output_dir, shape=(256, 256), max_height=None, dilation_radius=0):
    os.makedirs(output_dir, exist_ok=True)
    for shp_file in os.listdir(shp_path):
        if shp_file.endswith('.shp'):
            gdf = gpd.read_file(os.path.join(shp_path, shp_file))
            gdf['height'] = gdf['height'].apply(lambda x: x if not math.isnan(x) else 0)
            actual_max_height = int(gdf['height'].max()) if not gdf.empty else 0
            use_max_height = max_height if max_height is not None else actual_max_height
            xmin, ymin, xmax, ymax = gdf.total_bounds if not gdf.empty else (0, 0, 1, 1)
            transform = from_bounds(xmin, ymin, xmax, ymax, shape[1], shape[0])

            # 定义结构元素（8邻域）
            structure = generate_binary_structure(2, 1)

            for height_level in range(1, use_max_height + 1):
                if height_level > actual_max_height:
                    img_data = np.full((shape[0], shape[1]), 0, dtype=np.uint8)
                else:
                    height_filter = (gdf['height'] >= height_level)
                    raster = rasterio.features.rasterize(
                        [(geom, 1) for geom in gdf[height_filter].geometry],
                        out_shape=shape,
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )

                    # 二值图（0/255）
                    img_data = np.zeros((shape[0], shape[1]), dtype=np.uint8)
                    img_data[raster == 1] = 255

                    # 对掩膜膨胀处理
                    if dilation_radius > 0:
                        binary_mask = (img_data > 0)
                        for _ in range(dilation_radius):
                            binary_mask = binary_dilation(binary_mask, structure=structure)
                        img_data = (binary_mask * 255).astype(np.uint8)

                output_filename = f"{os.path.splitext(shp_file)[0]}_{height_level}.png"
                Image.fromarray(img_data).save(
                    os.path.join(output_dir, output_filename)
                )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Beijing shapefiles into voxel-slice PNG maps.")
    parser.add_argument("--shp-path", default=shp_path)
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--max-height", type=int, default=50)
    parser.add_argument("--dilation-radius", type=int, default=0)
    args = parser.parse_args()

    if args.dilation_radius > 0:
        shp_to_pixel_dilation(
            args.shp_path,
            args.output_dir,
            max_height=args.max_height,
            dilation_radius=args.dilation_radius,
        )
    else:
        shp_to_pixel(args.shp_path, args.output_dir, max_height=args.max_height)

