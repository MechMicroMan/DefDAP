# Docs README #

The documentation for the DefDAP is generated using the Sphinx Python package.

Documentation is written in reStructuredText format (in the source folder) and compiled by the Sphinx package into a HTML source (in the build/html folder). At compilation time Sphinx also scrapes docstrings from the Python scripts and adds these to the documentation.

Requrements to compile documentation
======================================

Install Sphinx (http://www.sphinx-doc.org/en/master/) and the theme:

`conda install sphinx sphinx-rtd-theme`


Compiling documentation
=========================

Then, autogenerate the API documentation by running the following command from the docs folder:

`make buildapi`

To clean the build folder and start fresh:

`make clean`

Then to recompile the html documentation, use the following command from the docs folder:

`make html`

TODO: Once a commit is made to the master branch the docs folder is mirrored to mechmicroman.github.io/defdap, providing an online version of the documentation.
