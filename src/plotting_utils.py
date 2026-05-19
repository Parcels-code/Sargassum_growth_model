from matplotlib.colors import LinearSegmentedColormap

eps = 1e-9
biomass_change_cmap = LinearSegmentedColormap.from_list(
    "red_orange_green",
    [
        (0.0, "#8b0000"),
        (0.2 - eps, "#ff6600"),  # orange up to just below 0.2
        (0.2, "#90ee90"),        # exactly 0.2 is green
        (1.0, "#006400"),
    ],
)
