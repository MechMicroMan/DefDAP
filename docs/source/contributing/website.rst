Documentation
===========================

The narrative documentation on this website (including this page) is built from source files which can be found in the ``docs/source`` directory.
It is automatically built to https://defdap.readthedocs.io/en/latest/ when changes are merged to the main branch.
Documentation for the develop branch is available at https://defdap.readthedocs.io/en/develop/.

Writing documentation
-----------------------

DefDAP uses Sphinx to build the documentation, which is written in reStructuredText format.
More details about the format can be found in the Sphinx documentation: 

https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html.

Building documentation
-----------------------

To build the documentation yourself, you will first need to install the dependencies for building the documentation.
You can do this using pip:

``pip install defdap[docs]``

Then, you can build the documentation using the following command from the ``docs/source`` directory:

``make docs``

The built html documentation will be available in the ``docs/build/html`` directory, and can be opened in a web browser.
