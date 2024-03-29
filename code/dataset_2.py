import os
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
import math
from PIL import Image


def split_tif(folder_path, file_name, block_size):

    # Open the original TIFF file
    tif_file = gdal.Open(os.path.join(folder_path, file_name))

    # Get the geotransform information
    geotransform = tif_file.GetGeoTransform()
    xmin = geotransform[0] # TM2 (m)
    ymax = geotransform[3] # TM2 (m)
    res = geotransform[1] # meters/pixel

    # Get the raster size
    x_size = tif_file.RasterXSize # pixels
    y_size = tif_file.RasterYSize # pixels
    xlen = res * x_size # meters
    ylen = res * y_size # meters

    # Define the size of the blocks
    # block_size = 5 # meters by meters

    test_1 = 0
    test_2 = 0

    # Iterate over the bounding boxes of each 1m x 1m block
    for y in np.arange(ymax, ymax - y_size * res, -block_size):
        for x in np.arange(xmin, xmin + x_size * res, block_size):
            # Calculate the coordinates of the block
            x_min_block, y_max_block = x, y
            x_max_block, y_min_block = x + block_size, y - block_size
            block_coord = [x + block_size/2, y - block_size/2]

            # Read the pixel values for the block
            pixel_values = tif_file.ReadAsArray(
                int((x - xmin) / res), int((ymax - y) / res), int(block_size / res), int(block_size / res))
            
            if pixel_values is None:
                # print("Error: Failed to read pixel values.", test_1)            
                # print(int(block_coord[0]), int(block_coord[1]), test_1)
                # test_1 = test_1 + 1
                continue  # Skip processing this block and continue to the next one
            
            # Create a new dataset for the 1m x 1m area
            driver = gdal.GetDriverByName('GTiff')

            if driver is None:
                print("Error: Failed to get GDAL driver.")
                exit(1)        
            
            out_tif = driver.Create(
                os.path.join(folder_path, f"{block_size}m_{block_size}m", f"{int(block_coord[0])}_{int(block_coord[1])}.tif"),
                int(block_size/res), int(block_size/res), tif_file.RasterCount, gdal.GDT_Byte
            )

            if out_tif is None:
                # print("Error: Failed to create output TIFF dataset.")
                # exit(1)
                print("Error: Failed to create output TIFF dataset.", test_2)
                test_2 = test_2 + 1
                continue  # Skip processing this block and continue to the next one
            
            # Set the geotransform
            out_tif.SetGeoTransform((x_min_block, res, 0, y_max_block, 0, -res))
            srs = osr.SpatialReference()            # establish encoding
            srs.ImportFromEPSG(3826)                # TWD97 lat/long
            out_tif.SetProjection(srs.ExportToWkt()) # export coords to file

            # Write the pixel values to the new dataset
            for band_num in range(tif_file.RasterCount):
                band_values = pixel_values[band_num]
                out_tif.GetRasterBand(band_num + 1).WriteArray(band_values)
            
            # write to disk
            out_tif.FlushCache()

            # Close the new dataset
            out_tif = None

    # Close the original TIFF file
    tif_file = None

def count_pixels(image):
    """
    Count white and black pixels in the image.
    """
    width, height = image.size
    white_count = 0
    black_count = 0
    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if pixel == (255, 255, 255, 255):  # Assuming white pixels are (255, 255, 255)
                white_count += 1
            elif pixel == (0, 0, 0, 0):       # Assuming black pixels are (0, 0, 0)
                black_count += 1
    return white_count, black_count

def delete_if_mostly_white_or_black(image_path):
    """
    Delete the image if it contains more than half white or black pixels.
    """
    image = Image.open(image_path)
    white_count, black_count = count_pixels(image)
    total_pixels = image.width * image.height
    if white_count > total_pixels / 2 or black_count > total_pixels / 2:
        os.remove(image_path)
        print(f"Deleted {image_path}")

def main(folder_path):
    """
    Main function to iterate through TIFF files in the folder.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tif") or file_name.endswith(".tiff"):
            file_path = os.path.join(folder_path, file_name)
            delete_if_mostly_white_or_black(file_path)


# list = [76, 78, 79, 82]
# for i in list:
#     folder_path = f"D:/Yehmh/test_py/202301/P000{i}/"
#     file_name = f"202301P000{i}_RGB_transparent_mosaic_group1.tif"
#     split_tif(folder_path, file_name, 5)
#     # os.makedirs(os.path.join(folder_path, "5m_5m"))

# # clean data
# list = [70, 71, 75, 76, 78, 79, 82]
# for i in list:
#     folder_path = f"D:/Yehmh/test_py/202301/P000{i}/5m_5m"
#     main(folder_path)

# folder_path = "D:/Yehmh/test_py/202301/P00074_transect_1/"
# file_name = "202301P00074_RGB_transparent_mosaic_group1.tif"
# split_tif(folder_path, file_name, 5)

folder_path = "D:/Yehmh/test_py/202301/P00074_transect_1/5m_5m"
main(folder_path)
folder_path = "D:/Yehmh/test_py/202301/P00073_transect_234/5m_5m/unknown"
main(folder_path)