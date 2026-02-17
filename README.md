# Sargassum growth model

Repository to do simulations, post-processing and exploratory analyses of the MSc project **Integrating physical transport and biological growth of *Sargassum* in a Lagrangian framework**.
The specific files of the thesis' figures are listed below.
Note that this repository also contains older simulations files. 

## Simulation notebook

The `NOTEBOOK.ipynb` file guides you through (part of) the process of modelling growth of *Sargassum* rafts during transport in the Atlantic Ocean. 
It specifically contains a customizable *Sargassum* growth model that can be included in Lagrangian simulations using Parcels. 
Variables as maximum growth rate and optimal temperature can therefore be easily adapted in this growth model. The notebook is build up by the following steps:
- Defining the limitation functions on growth by physico-chemical factors
- Creating a map of Sargassum locations based on satellite detections from the Sargassum Watch System (SaWS)
- Combining all the required data fields for *Sargassum* simulation as a fieldset
- Defining *Sargassum* rafts as particles for a simulation with related characteristic variables
- Setting up the operational kernels to describe *Sargassum* behaviour
- Executing the simulation
- Visualizing the output

## Overview thesis figures
The following list indicates in which files the output processing and visualization of the thesis figures can be found.
For the results, the specific folder in `data_Elena` on Lorenz where certain particlesets (used in notebook) can be found is also added.

### Background

- **Figure 2.2:** `LIMITATION_ANALYSIS.ipynb` 

### Methods

- **Figure 3.1:** `check_stokes_kernels.ipynb` 
- **Figure 3.2:** `LIMITATION_ANALYSIS.ipynb` 
- **Figure 3.3:** `grid_initialization_notebook.ipynb`
  
### Results
- **Figure 4.1:** `check_feedbacks.ipynb` & particlesets in `Satellite_out` and `FINAL`
- **Figure 4.2:** `OUTPUT_POSTPROCESSING/Output_kN.ipynb` & particlesets in `Satellite_out`
- **Figure 4.3:** `OUTPUT_POSTPROCESSING/Output_kN.ipynb` & particlesets in `Satellite_out`
- **Figure 4.4:** `OUTPUT_POSTPROCESSING/Output_kN.ipynb` & particlesets in `Satellite_out`
- **Figure 4.5:** `OUTPUT_POSTPROCESSING/Output_kN.ipynb` & particlesets in `Satellite_out`
- **Figure 4.6:** `OUTPUT_POSTPROCESSING/Output_kN.ipynb` & particlesets in `Satellite_out`
- **Figure 4.7:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure 4.8:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure 4.9:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure 4.10:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure 4.11:** `LIMITATION_ANALYSIS.ipynb`
- **Figure 4.12:** `check_feedbacks.ipynb` & particlesets in `Satellite_out` and `FINAL`

### Appendices
- **Figure A.1:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure A.2:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure A.3:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure B.1:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
- **Figure C.1:** `OUTPUT_POSTPROCESSING/Output_12months.ipynb` & particlesets in `FINAL`
