# Docs README #

The documentation for the DefDAP is generated using the Sphinx Python package.

Documentation is written in reStructuredText format and compiled by the Sphinx package into a HTML source. At compilation time Sphinx also scrapes docstrings from the Python scripts and adds these to the documentation.

Requrements to compile documentation
======================================

Install: Sphinx (http://www.sphinx-doc.org/en/master/) - (conda install sphinx/pip install sphinx)


Compiling documentation
=========================

First, autogenerate the API documentation by running the following command::

sphinx-apidoc ../defdap -o ./source -f

Then to recompile the html documentation, use the following command (run from the docs folder)::

sphinx-build -b html ./source ./

Once a commit is made to the master branch the docs folder is mirrored to mehcmicroman.github.io/defdap, providing an online version of the documentation.