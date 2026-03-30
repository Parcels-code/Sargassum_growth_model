import numpy as np
import xarray as xr

from PIL import Image
from scipy.ndimage import binary_dilation

def sarg_grid_from_sat(image_name, north, south, east, west, coarse=False):

    #Loading the image as an RGB array
    img = Image.open(image_name).convert("RGB")
    img_array = np.array(img)

    #Computing brightness by approximating average or weighted sum of RGB
    brightness = img_array.mean(axis=2)

    #Creating a land mask based on the brown color of landmass
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

    #Brown tends to be dark, reddish, and not too saturated
    land_mask = (
        (r > 60) & (r < 160) &          # moderate red
        (g > 30) & (g < 110) &          # moderate green
        (b < 70) &                      # low blue
        (brightness < 120) &            # exclude bright oranges
        ((r - g) > 15) &                # red clearly higher than green
        ((r - b) > 40)                  # red much higher than blue
        )
    #Expanding land mask with binary_dilation method by 20 pixels (~20 km as 1 pixel ~ 1 km resolution)
    expanded_land_mask = binary_dilation(land_mask, iterations=20)

    #Setting threshold to get binary mask - set at 60 to include bright pixels and especially also red pixels
    threshold = 60
    binary_mask = (brightness > threshold).astype(int)

    #Applying the expaned land mask on binary mask
    binary_mask[expanded_land_mask] = 0

    #Creating coordinate grids based on bounding boxes
    height, width = binary_mask.shape
    lats = np.linspace(north, south, height)
    lons = np.linspace(west, east, width)

    #Creating 2D coordinate grids
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    #Downsample with stride 2 (or other number) if you want to select less particles
    stride = 2 if coarse else 1
    mask = binary_mask[::stride, ::stride]
    lat = lat_grid[::stride, ::stride]
    lon = lon_grid[::stride, ::stride]

    amount = int(mask.sum())

    #Creating 2D grids of Sargassum release locations
    sarg_lon_grid = xr.DataArray(lon).where(mask == 1)
    sarg_lat_grid = xr.DataArray(lat).where(mask == 1)

    #To prepare the grids as ParticleSet input, NaNs are removed and and arrays are ravelled (flattened)
    no_nan_mask = (~np.isnan(sarg_lon_grid)) & (~np.isnan(sarg_lat_grid))
    sarg_lon_grid = sarg_lon_grid.values[no_nan_mask].ravel()
    sarg_lat_grid = sarg_lat_grid.values[no_nan_mask].ravel()

    return sarg_lon_grid, sarg_lat_grid, amount