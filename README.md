<p align="center">
  <img src="docs/source/_static/logo.png">
</p>

<h4 align="center">A python library for correlating EBSD and HRDIC data.</h4>

<p align="center">
  <img alt="Release" src="https://img.shields.io/github/v/release/MechMicroMan/DefDAP?include_prereleases">
  <img alt="Python 3.7" src="https://img.shields.io/badge/python-3.7-red">
  <img alt="License" src="https://img.shields.io/github/license/MechMicroMan/DefDAP">
  <a href="https://mybinder.org/v2/gh/MechMicroMan/DefDAP/master?filepath=example_notebook.ipynb">
    <img alt="Binder" src="https://mybinder.org/badge_logo.svg">
  <a href="https://zenodo.org/record/3688096">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.3688096.svg">
    
  </a>
</p>

<p align="center">
  <a href="#how-to-install">How to install</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#credits">Credits</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>


## How to install

- Download and install Anaconda from https://repo.continuum.io/archive/ (Anaconda3-2020.02 has been tested)

- Download the DefDAP package using the 'Clone or download' button above, and extract to a directory

- Open a terminal in that directory and run `pip install -e .`

## How to use

- To start the example notebook, use the command `jupyter lab` and click the notebook (.ipynb file)

- Try out the example notebook using the binder link above (Note: change the matplotlib plotting mode to inline, some of the interactive features will not work)

- Here is a video demonstrating the basic functionality of the package: 
[DefDAP demo](http://www.youtube.com/watch?v=JIbc7F-nFSQ "DefDAP demo")

## Documentation

- For more help, see the documentation under /docs/html

## Contributing

- For information about contributing see the [guidelines](/docs/contributing.md)

## Credits

The software uses the following open source packages:

- [scipy](http://scipy.org/)
- [numpy](http://numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-image](http://scikit-image.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [networkx](https://networkx.github.io/)
- [pandas](http://pandas.pydata.org)
- [peakutils](https://peakutils.readthedocs.io/en/latest/)
- [matplotlib_scalebar](https://pypi.org/project/matplotlib-scalebar/)
- [IPython](https://ipython.org/)
- [jupyter](https://jupyter.org/)

## License

[Apache 2.0 license](/LICENSE)
