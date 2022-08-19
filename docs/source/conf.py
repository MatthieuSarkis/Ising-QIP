# pip install sphinx
# pip install furo      (it is a theme)
# mkdir docs
# cd docs
# sphinx-quickstart
# sphinx-apidoc ../src/ -o ./source/apidoc/

# comment out all the
# Submodules
# ----------
# and
# Subpackages
# ----------
# for clarity of the API tree

# OPTIONALLY INCLUDE THE README
# pip install m2r2     (to translate .md to .rst)
# m2r2 ../README.md ./source/
# change location of images in README.rst -> ./_static/images/
# on top of index.rst add '.. include:: ./README.md'



# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ising-QIP'
copyright = '2022, Matthieu Sarkis'
author = 'Matthieu Sarkis'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # to build documentation from docstrings.
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon', # to handle docstrings in written in google style.
    'sphinx.ext.viewcode', # to have a link to the source code for each definition.
    'm2r2', # to translate .md to .rst (to include the README.md in the beginning of index.rst)
]

autodoc_default_options = {
    'members': True,
    # The ones below should be optional but work nicely together with
    # example_package/autodoctest/doc/source/_templates/autosummary/class.rst
    # and other defaults in sphinx-autodoc.
    'show-inheritance': True,
    'inherited-members': True,
    'no-special-members': True,
}

#ignore_warnings = [("code/api/qml_transforms*", "no module named pennylane.transforms")]

# Add any paths that contain templates here, relative to this directory.
#templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = "index"

# today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pip install furo
html_theme = 'furo'
html_static_path = ['_static']