from matplotlib.colors import LinearSegmentedColormap

def biomass_change_cmap(vmax):
    eps = 1e-9

    split = 100./vmax
    return LinearSegmentedColormap.from_list(
        "red_orange_green",
        [
            (0.0, "#8b0000"),
            (split - eps, "#ff6600"),  # orange up to just below 1.0
            (split, "#90ee90"),        # exactly 1.0 is green
            (1.0, "#006400"),
        ],
    )
