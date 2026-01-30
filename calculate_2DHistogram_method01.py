#PROBABILITY DENSITY FUNCTION

#This code bins the particle positions into histograms at each timestep using np.histogram2D. The result is a heat map of the cumulative particle density over time which, when normalised by the total number of particle positions, yields a probability map.

#Author: Jimena Medina Rubio

#Created on: 18/03/2023

#0. Imports and package versions
import parcels
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean
import cmocean.cm as cmo
import pandas as pd

#. DEFINITION OF FUNCTIONS

def probability_density(ds, bins_x, bins_y):
    
    """
    Calculates the PDF of the longitude and latitude coordinates of the trajectories at each observation.
    
    Input variables
    -ds: OceanParcels output of lon & lat of each particle at each timestep
    -bins_x & bins_y: number of bins in x/y direction 
    """

    def histogram(lon, lat, bins_x, bins_y): 
        
        #define the coordinates of the edges of the bins
        bins_edges_x = np.histogram_bin_edges(lon, bins=bins_x)
        bins_edges_y = np.histogram_bin_edges(lat, bins=bins_y)
        
        #calculate the 2D normalised histogram & bin edges
        H, x, y = np.histogram2d(lon.flatten(), lat.flatten(), bins=[bins_edges_x, bins_edges_y], density=True)
        return H, x, y
    
    #apply histogram function to all trajectories at every observation
    result = xr.apply_ufunc(
        histogram,
        ds['lon'].values,
        ds['lat'].values,
        bins_x,
        bins_y,
        input_core_dims=[['traj', 'obs'], ['traj', 'obs'], [], []],
        output_core_dims=[['binx', 'biny'], [], []],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[float])
    
    #define the bin centres from the output bin edges
    bins_centres_x=np.linspace(result[1][0], result[1][-1], len(result[1])-1)
    bins_centres_y=np.linspace(result[2][0], result[2][-1], len(result[2])-1)
    
    #convert particle counts per grid cell into a data array
    da_result = xr.DataArray(result[0], 
                             dims=['binx', 'biny'], 
                             coords={'binx': bins_centres_x, 'biny': bins_centres_y}, name='%') 
    
    #set values equal to zero to NaN & normalises results so that sum of probability =100
    da_result = da_result.where(da_result != 0, np.nan)*100/np.nansum(da_result)
    
    return da_result.T

def probability_map(probability, xlim, ylim, title, da_velocity):
    
    """ All-included plot of the desired domain specified by xlim & ylim """
    
    fig=plt.figure(figsize=(13,6)) 
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0.0))
    
    #create grid
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #choose: plot coastlines or velocity field stored in data array
    if np.all(da_velocity is None):
        ax.coastlines(resolution='10m')
        ax.add_feature(cartopy.feature.LAND, facecolor='grey')
    else:   
        da_velocity.plot(ax=ax, cmap=cmo.balance, alpha=0.7)

    gl = ax.gridlines(draw_labels=['left','bottom'], zorder=1, alpha=0.3, linestyle='--')

    #plotting probability results 
    probability.plot(ax=ax, cmap=cmo.matter)
    plt.title(title)
                
    return plt.show()



