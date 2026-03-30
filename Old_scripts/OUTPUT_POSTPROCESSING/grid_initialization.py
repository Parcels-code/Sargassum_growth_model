import math
import xarray as xr
import numpy as np

def create_grid_mask(zonal_velocity_file: str, lonmin: float, latmin: float, 
                     lonmax: float, latmax: float, depth: float, 
                     direction: str, outputfile: str) -> xr.DataArray:
    
    lon_name = 'nav_lon'  
    lat_name = 'nav_lat'  
     
    ds = xr.open_dataset(zonal_velocity_file) 
    
    #Selecting the velocity component at the nearest depth level to 'depth'
    dv = ds['vozocrtx'].sel(deptht=depth, method='nearest') #'vozocrtx' is the zonal velocity
    
    #Printing shape of the coordinate arrays
    print(ds['nav_lon'].shape, ds['nav_lat'].shape)

    #Extracting the longitude and latitude arrays
    longrid = ds[lon_name]
    latgrid = ds[lat_name]

    #Create an array full of ones, same shape as the longitude/latitude grid
    #This will serve as the initial mask (all points "valid")
    mask = xr.DataArray(
        np.ones_like(longrid),  # fill array with ones
        dims=longrid.dims,      # use same dimensions as longrid (e.g., lat, lon)
        coords={lon_name: longrid, lat_name: latgrid}  # attach coordinates
    )

    # Modify the mask based on missing velocity values depending on 'direction'
    if direction == 'up':
        # For 'up', keep points where velocity is missing (NaN), set others to NaN
        mask = mask.where(dv.isnull(), np.nan)
    elif direction == 'down':
        # For 'down', keep points where velocity exists, set missing points to NaN
        mask = mask.where(~dv.isnull(), np.nan)
    else:
        # If direction is not recognized, raise an error
        raise ValueError(f'{direction} should be up or down!')

    #Also selecting specific longitude/latitude domain and setting points outside bounds to NaN
    mask = mask.where(
        (longrid >= lonmin) & (longrid <= lonmax) &  #longitude within domain
        (latgrid >= latmin) & (latgrid <= latmax),   #latitude within domain
        np.nan  )

    #If an output file path is given, saving the mask as NetCDF file
    if outputfile:
        mask.to_netcdf(outputfile)
        print(f"Landmask is saved as: {outputfile}")

    #Returning the final mask DataArray
    return mask

def grid_creation(lon_min, lon_max, lat_min, lat_max, landmask: xr.DataArray,
                      nlon=12, nlat=12, lon_name='lon', lat_name='lat'):
    """
    Creating a regular grid of lon/lat points for a fieldset with resolution of 1/12 deg.
    Removing points where landmask == 1.
    """
    #Creating evenly spaced coordinates between minimum and maximum lon and lat and nlon/nlat particles per degree
    release_lon = np.linspace(lon_min, lon_max, ((lon_max-lon_min)*nlon + 1))
    release_lat = np.linspace(lat_min, lat_max, ((lat_max-lat_min)*nlat + 1))

    #Creating meshgrid and flatten
    lon_grid, lat_grid = np.meshgrid(release_lon, release_lat)
    
    #Converting to 1D arrays for vectorized sampling
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    #Extracting the nav_lon/nav_lat arrays
    nav_lon = landmask['nav_lon'].values
    nav_lat = landmask['nav_lat'].values

    #Using scipy’s griddata as interpolater for 2D coordinate fields
    from scipy.interpolate import griddata

    #Combining the 2D longitude and latitude grids into a single array of coordinate pairs (lon, lat)
    points = np.column_stack((nav_lon.ravel(), nav_lat.ravel()))

    #Flattening the landmask array to match the flattened coordinate pairs
    landmask_values = landmask.values.ravel()

    #Interpolating the landmask data onto the target grid defined by (lon_flat, lat_flat)
    landmask_interp = griddata(points, landmask_values, (lon_flat, lat_flat), method='nearest')

    #Creating a mask that only selects suitable points in the ocean () and apply it on lon and lat dataset to create a grid in the ocean
    grid_mask = (landmask_interp != 1)     #!= 1 (means is not 1 aka not land)
    lon_grid = lon_flat[grid_mask]
    lat_grid = lat_flat[grid_mask]

    return lon_grid, lat_grid


def sarg_grid_from_sat(image_name, north, south, east, west, coarse=False):

    #Loading the image as an RGB array
    from PIL import Image
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
    from scipy.ndimage import binary_dilation
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

    #Downsample with stride 2 if you want to select less particles
    stride = 2 if coarse else 1
    mask = binary_mask[::stride, ::stride]
    lat = lat_grid[::stride, ::stride]
    lon = lon_grid[::stride, ::stride]

    amount = int(mask.sum())

    #Creating 2D grids of Sargassum release locations
    sarg_lon_grid = xr.DataArray(lon).where(mask == 1)
    sarg_lat_grid = xr.DataArray(lat).where(mask == 1)
    print('Shape of grid:', np.shape(sarg_lon_grid))

    #To prepare the grids as ParticleSet input, NaNs are removed and and arrays are ravelled (flattened)
    no_nan_mask = (~np.isnan(sarg_lon_grid)) & (~np.isnan(sarg_lat_grid)) 
    sarg_lon_grid = sarg_lon_grid.values[no_nan_mask].ravel()
    sarg_lat_grid = sarg_lat_grid.values[no_nan_mask].ravel()
    print('Reshaped grid as particle set:', np.shape(sarg_lon_grid))

    return sarg_lon_grid, sarg_lat_grid, amount