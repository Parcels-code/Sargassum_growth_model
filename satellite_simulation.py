import os
from datetime import datetime
import numpy as np

import parcels

import src.load_copernicus_fieldset as load_copernicus_fieldset  # noqa: E402
from src.saws_functions import release_points_from_SaWS_images  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402

os.makedirs("Simulations", exist_ok=True)

all_exps = []
default_dict = {
    "k_N": 0.001,
    "k_P": 0.0008,
    "month": 7,
    "with_Stokes": 1,
    "wind_coeff": 0,
    "z_lower": 1.0,
    "runtime": 31,
}

for runtime in [31, 60, 90]:
    for z_lower in [1, 0.5, 0.25, 0, "surface"]:
        for wind_coeff in [0.0, 0.01, 0.02, 0.03]:
            new_dict = default_dict.copy()
            new_dict["z_lower"] = z_lower
            new_dict["wind_coeff"] = wind_coeff
            new_dict["runtime"] = runtime
            all_exps.append(new_dict)

for month in [7, 10, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12]:
    new_dict = default_dict.copy()
    new_dict["month"] = month
    all_exps.append(new_dict)

for k_N in [0.001, 0.000129, 0.01]:
    new_dict = default_dict.copy()
    new_dict["k_N"] = k_N
    all_exps.append(new_dict)

for k_P in [0.0008, 0.8, 0.008]:
    new_dict = default_dict.copy()
    new_dict["k_P"] = k_P
    all_exps.append(new_dict)

print(f"Total number of experiments: {len(all_exps)}")

for exp in all_exps:
    startmonth = f"2024-{exp['month']:02d}"
    startdate = np.datetime64(f"{startmonth}-01T00:00:00")
    enddate = startdate + np.timedelta64(exp["runtime"], "D")
    filename = f"Simulations/Simulation_Satellite_{startmonth}_kN_{exp['k_N']}_kP_{exp['k_P']}_zlower_{exp['z_lower']}_wind_{exp['wind_coeff']:.2f}_runtime_{exp['runtime']}.parquet"

    if os.path.exists(filename):
        # Skip if the simulation has already been run
        continue

    fieldset = load_copernicus_fieldset.create_fieldset(startdate=startdate, enddate=enddate)

    # Model parameters
    fieldset.z_upper = 0.0                    # Upper depth extent of Sargassum raft [m]
    fieldset.z_lower = exp["z_lower"]         # Lower depth extent of Sargassum raft [m]
    fieldset.wind_coeff = exp["wind_coeff"]   # Windage coefficient [fraction]
    fieldset.mu_max = 0.095                   # Maximum growth rate [doublings/day]
    fieldset.mort = 0.025                     # Mortality rate [loss/day]
    fieldset.T_min = 20.0                     # Minimum temperature [degC]
    fieldset.T_opt = 27.5                     # Optimal temperature [degC]
    fieldset.T_max = 31.0                     # Maximum temperature [degC]
    fieldset.S_opt = 36.0                     # Optimal salinity [psu]
    fieldset.k_N = exp["k_N"]                 # Nitrogen uptake half saturation [mmol/m3]
    fieldset.k_P = exp["k_P"]                 # Phosphorus uptake half saturation [mmol/m3]

    coords, _ = release_points_from_SaWS_images(startdate.astype(datetime), stride=8)

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=SargassumParticle,
        x=coords[:, 0],
        y=coords[:, 1],
        z=np.zeros_like(coords[:, 0]),
        t=np.datetime64(f'{startmonth}-01T00:00:00'),
    )

    # Do an initial sampling of the Biological Growth Model
    # to initialize the particle variables
    sargassum_kernels.SargassumBiologicalGrowthModel(pset, fieldset)

    # Now load datasets into memory for faster execution
    for model in fieldset.models:
        model.data.load()
    fieldset.describe()

    pfile = parcels.ParticleFile(
        filename,
        outputdt=np.timedelta64(2, 'h'),
    )
    fieldset.output_file = pfile  # for writing stranded particles

    kernels = [
        sargassum_kernels.Stranding,
        parcels.kernels.AdvectionRK2,
        sargassum_kernels.DepthIntegratedStokesDriftRK2,
        sargassum_kernels.WindageRK2,
        sargassum_kernels.SargassumBiologicalGrowthModel,
        sargassum_kernels.DeleteOutOfBounds,
        sargassum_kernels.DeleteInterpolationError,
    ]

    if exp["wind_coeff"] == 0.0:
        kernels.remove(sargassum_kernels.WindageRK2)

    if exp["z_lower"] == 0:
        kernels.remove(sargassum_kernels.DepthIntegratedStokesDriftRK2)

    if exp["z_lower"] == "surface":
        kernels.remove(sargassum_kernels.DepthIntegratedStokesDriftRK2)
        kernels.insert(2, sargassum_kernels.SurfaceStokesDriftRK2)

    pset.execute(
        kernels=kernels,
        runtime=np.timedelta64(exp["runtime"], 'D'),
        dt=np.timedelta64(10, 'm'),
        output_file=pfile,
    )
