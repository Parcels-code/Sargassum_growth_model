import os

import numpy as np

import parcels

import src.load_copernics_fieldset as load_copernics_fieldset  # noqa: E402
from src.saws_functions import sarg_grid_from_sat  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402

start_images = [
    {
        'image_name': "../SaWSdata/C20241772024183.1KM.C_ATLANTIC.7DAY.L3D.FA_UNET_DENSITY.png",
        'bbox': [22.0, 0.0, -63.0, -38.0],
    },
    {
        'image_name': "../SaWSdata/C20241772024183.1KM.CE_ATLANTIC.7DAY.L3D.FA_UNET_DENSITY.png",
        'bbox': [22.0, 0.0, -38.0, -11.5],
    }
]
release_lon, release_lat, _ = sarg_grid_from_sat(start_images, coarse=True)

fieldset = load_copernics_fieldset.create_fieldset(startmonth="2024-07")

#TODO these can be removed as fieldset constants
fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-1]
# #Nitrogen half saturation constant
fieldset.add_constant('k_N', 0.001) #mmol/m3
#Overall maximal growth rate (Corbin & Oxenford)
fieldset.add_constant('MGR_SF3', 0.124)
fieldset.add_constant('MGR_SN1', 0.083)
fieldset.add_constant('MGR_SN8', 0.053)
#Set initial weight
fieldset.add_constant('initial_weight', 50) #grams

for k_N in [0.001, 0.000129, 0.01]:
    filename = f"Simulations/Simulation_Satellite_kN_{k_N}.parquet"

    fieldset.k_N = k_N

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass = SargassumParticle,
        lon = release_lon,
        lat = release_lat,
        z = np.zeros_like(release_lon),
        time = np.datetime64('2024-07-01T00:00:00'),
    )

    pfile = parcels.ParticleFile(
        filename,
        outputdt=np.timedelta64(2, 'h'),
    )

    kernels = [
        parcels.kernels.AdvectionRK4,
        sargassum_kernels.di_Stokes_drift,
        sargassum_kernels.windage_drift,
        sargassum_kernels.sargassum_biological_growth_model,
        sargassum_kernels.stranding,
        sargassum_kernels.DeleteOutOfBounds,
    ]

    if not os.path.exists(filename):
        pset.execute(
            kernels,
            runtime=np.timedelta64(31, 'D'),
            dt=np.timedelta64(10, 'm'),
            output_file=pfile,
    )
