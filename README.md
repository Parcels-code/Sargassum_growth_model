# Sargassum growth model

Repository for the interactive notebook and simulations for the article **A Lagrangian framework to simulate Sargassum transport and growth** by Elena Gianotten, Meike F. Bos, Darshika Manral, Fabio Nauer, Erik Zettler, Linda
A. Amaral-Zettler, and Erik van Sebille

## How to install

1. Install [Pixi](https://pixi.sh/)
- On Mac or Linux, run the following command in your terminal:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
- On Windows, run the following command in PowerShell:
```
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```
2. Restart your terminal
3. Clone this repository
```bash
git clone --depth 1 https://github.com/Parcels-code/Sargassum_growth_model.git
```
4. Change into the repository directory
```bash
cd Sargassum_growth_model
```
5. Install the required dependencies
```bash
pixi install
```
6. Start JupyterLab
```bash
pixi run jupyter lab
```
7. The `interactive_sargassum_notebook.ipynb` opens automatically. Run the cells to see the simulations.

## Content of this repository
- `interactive_sargassum_notebook.ipynb`: The interactive notebook to run the Sargassum growth simulations and visualize the results.
- `src/`: A directory containing the source code for the Sargassum growth model, including the transport and growth kernels in `src/sargassum_kernels.py`.
- `satellite_simulation.py`: The script that runs the simulation described in the manuscript with satellite-based initialisation.
- `basin_simulation.py`: The script that runs the simulation described in the manuscript with basin-wide initialisation.
- `Manuscript_Figures/`: A directory containing the notebooks used to create the figures for the manuscript.
