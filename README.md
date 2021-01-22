<p align="center">
  <img src="http://mechmicroman.github.io/DefDAP/_images/logo.png">
</p>

<h4 align="center">A python library for correlating EBSD and HRDIC data.</h4>

<p align="center">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/defdap">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/defdap">
  <img alt="License" src="https://img.shields.io/github/license/MechMicroMan/DefDAP">
  <a href="https://defdap.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/defdap/badge/?version=latest" alt='Documentation Status' />
  </a>
  <a href="https://mybinder.org/v2/gh/MechMicroMan/DefDAP/master?filepath=example_notebook.ipynb">
    <img alt="Binder" src="https://mybinder.org/badge_logo.svg">
  </a>
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

- DefDAP can be installed from PyPI using the command `pip install defdap`

- If you use conda as your package manager (https://www.anaconda.com/), the prerequisite packages that are available from anaconda can be installed using the command `conda install scipy numpy matplotlib scikit-image scikit-learn pandas networkx jupyter ipython` (Anaconda3-2020.02 has been tested)

- For more information, see the [documentation](https://defdap.readthedocs.io/en/latest/installation.html)

## How to use

- To start the example notebook, use the command `jupyter lab` and click the notebook (.ipynb file)

- Try out the example notebook using the binder link above (Note: change the matplotlib plotting mode to inline, some of the interactive features will not work)

- Here is a video demonstrating the basic functionality of the package: 
[DefDAP demo](http://www.youtube.com/watch?v=JIbc7F-nFSQ "DefDAP demo")

## Documentation

- For more help, see the online [documentation](https://defdap.readthedocs.io/).

## Contributing

- For information about contributing see the [guidelines](https://defdap.readthedocs.io/en/latest/contributing.html)
- Any new functions or changes to arguments should be documented.

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

[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
