"""Sphinx configuration for the PolySwyft documentation site."""

import importlib.metadata

project = "PolySwyft"
author = "Kilian Scheutwinkel"
copyright = f"2025, {author}"

try:
    release = importlib.metadata.version("polyswyft")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

# Don't fail the build on imports we cannot satisfy on RTD.
autodoc_mock_imports = [
    "mpi4py",
    "pypolychord",
    "swyft",
    "torch",
    "pytorch_lightning",
    "wandb",
    "anesthetic",
    "lsbi",
    "cosmopower",
    "cmblike",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
typehints_fully_qualified = False
always_document_param_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"PolySwyft {release}"
