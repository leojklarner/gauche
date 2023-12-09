# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

# sys.path.insert(0, os.path.abspath("../../../gauche"))


# -- Project information -----------------------------------------------------

project = "GAUCHE"
copyright = "2022, Ryan Rhys-Griffiths"
author = "Ryan Rhys-Griffiths"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # "sphinx_copybutton",
    # "sphinx_inline_tabs",
    # "sphinxcontrib.gtagjs",
    # "sphinxext.opengraph",
    # "m2r2",
    # "nbsphinx",
    # "nbsphinx_link",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
nbsphinx_execute = "never"
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "Sphinx": ("https://www.sphinx-doc.org/en/stable/", None),
    "networkx": ("https://networkx.github.io/documentation/stable/", None),
    "nx": ("https://networkx.github.io/documentation/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

autodoc_default_options = {
    "special-members": "__init__",
}
