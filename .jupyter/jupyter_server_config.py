c = get_config()  # noqa: F821

# Open the project notebook by default when launching JupyterLab.
default_notebook_url = "/lab/tree/interactive_sargassum_notebook.ipynb"

# JupyterLab 4 uses LabApp/ExtensionApp for the landing URL.
c.LabApp.default_url = default_notebook_url

# Keep ServerApp aligned for non-Lab entry points.
c.ServerApp.default_url = default_notebook_url
