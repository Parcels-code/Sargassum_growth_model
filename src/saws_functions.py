import os
from datetime import datetime
import fnmatch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import numpy as np
import xarray as xr

from PIL import Image
from scipy.ndimage import binary_dilation


def sarg_grid_from_sat(date, stride=8):

    images = download_images(date)
    all_lons = []
    all_lats = []

    for image in images:
        #Loading the image as an RGB array
        img = Image.open(image).convert("RGB")
        img_array = np.array(img)

        coords_file = image.replace(".png", ".pgw")
        # Read world-file values
        with open(coords_file, "r", encoding="utf-8") as f:
            pixel_size_lon = float(f.readline().strip())   # pixel size in x (lon)
            _ = f.readline().strip()          # rotation (usually 0)
            _ = f.readline().strip()          # rotation (usually 0)
            pixel_size_lat = float(f.readline().strip())   # pixel size in y (lat, usually negative)
            upper_left_lon = float(f.readline().strip())   # lon of upper-left pixel center
            upper_left_lat = float(f.readline().strip())   # lat of upper-left pixel center

        # Coordinate vectors (pixel centers)
        lons = upper_left_lon + pixel_size_lon * np.arange(img.size[0])
        lats = upper_left_lat + pixel_size_lat * np.arange(img.size[1])

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

        #Creating 2D coordinate grids
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        #Downsample with stride 2 (or other number) if you want to select less particles
        mask = binary_mask[::stride, ::stride]
        lat = lat_grid[::stride, ::stride]
        lon = lon_grid[::stride, ::stride]

        #Creating 2D grids of Sargassum release locations
        sarg_lon = xr.DataArray(lon).where(mask == 1)
        sarg_lat = xr.DataArray(lat).where(mask == 1)

        #To prepare the grids as ParticleSet input, NaNs are removed and arrays are ravelled (flattened)
        no_nan_mask = (~np.isnan(sarg_lon)) & (~np.isnan(sarg_lat))
        all_lons.append(sarg_lon.values[no_nan_mask].ravel())
        all_lats.append(sarg_lat.values[no_nan_mask].ravel())

    coords = np.column_stack((np.concatenate(all_lons), np.concatenate(all_lats)))
    lon_bins = np.rint(coords[:, 0] / (stride*pixel_size_lon)).astype(np.int64)
    lat_bins = np.rint(coords[:, 1] / (stride*pixel_size_lat)).astype(np.int64)
    grid_bins = np.column_stack((lon_bins, lat_bins))
    _, unique_idx = np.unique(grid_bins, axis=0, return_index=True)
    unique_coords = coords[np.sort(unique_idx)]
    return unique_coords


def download_images(date):
    doy = date.timetuple().tm_yday
    year = date.year

    outdir = "SaWS_downloads"
    os.makedirs(outdir, exist_ok=True)

    regions = ["GOG", "C_ATLANTIC", "CE_ATLANTIC", "ECARIB", "PANAMA", "JAMAICA", "YUCATAN", "GCOOS"]
    images = []
    for region in regions:
        base = f"https://optics.marine.usf.edu/subscription/modis/{region}/{year}/comp/{doy}/"
        html = requests.get(base, timeout=30)
        html.raise_for_status()

        soup = BeautifulSoup(html.text, "html.parser")
        links = [urljoin(base, a["href"]) for a in soup.select("a[href]")]

        pattern = f"C*.1KM.{region}.7DAY.L3D.FA_UNET_DENSITY.png"
        for url in links:
            name = url.split("/")[-1]
            if fnmatch.fnmatch(name, pattern):
                out_path = os.path.join(outdir, name)
                images.append(out_path)

                if not os.path.exists(out_path):
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded {name} from SaWS server")

                pgw_name = name.replace(".png", ".pgw")
                pgw_path = os.path.join(outdir, pgw_name)
                if not os.path.exists(pgw_path):
                    url_pgw = f"https://optics.marine.usf.edu/cgi-bin/geo_reference?name=/{name}"
                    r = requests.get(url_pgw, timeout=60)
                    r.raise_for_status()
                    with open(pgw_path, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded {pgw_name} from SaWS server")
    return images
