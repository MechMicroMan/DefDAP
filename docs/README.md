# Docs README #

The documentation for the DefDAP is generated using the Sphinx Python package.

Documentation is written in reStructuredText format and compiled by the Sphinx package into a HTML source. At compilation time Sphinx also scrapes docstrings from the Python scripts and adds these to the documentation.

Requrements to compile documentation
======================================

Install: Sphinx (http://www.sphinx-doc.org/en/master/) - (conda install sphinx/pip install sphinx)


Compiling documentation
=========================

To recompile the documentation after a change, from the docs folder use the command::

sphinx-build -b html ./source ./html

If the structure of the Python scripts is significantly changed, api-doc can autogenerate the API documentation with the command::

sphinx-apidoc ../defdap -o ./source

Once a commit is made to the master branch the docs folder is mirrored to NOT YET IMPLEMENTED, providing an online version of the documentation.