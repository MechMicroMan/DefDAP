![DefDAP](https://defdap.readthedocs.io/en/latest/_images/logo.png)
#### A python library for correlating EBSD and HRDIC data.

[![PyPI version](https://img.shields.io/pypi/v/defdap "PyPI version")](https://pypi.org/project/defdap)
[![Supported python versions](https://img.shields.io/pypi/pyversions/defdap "Supported python versions")](https://pypi.org/project/defdap)
[![License](https://img.shields.io/github/license/mechmicroman/defdap "License")](https://github.com/MechMicroMan/DefDAP/blob/master/LICENSE)
[![Documentation status](https://readthedocs.org/projects/defdap/badge/?version=latest "Documentation status")](https://defdap.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg "Try on binder")](https://mybinder.org/v2/gh/MechMicroMan/DefDAP/master?filepath=example_notebook.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3688096.svg "DOI")](https://doi.org/10.5281/zenodo.3688096)

[How to install](#how-to-install "How to install") •
[How to use](#how-to-use "How to use") •
[Documentation](#documentation "Documentation") •
[Credits](#credits "Credits") •
[Contributing](#contributing "Contributing") •
[License](#license "License")

## How to install
- DefDAP can be installed from PyPI using the command `pip install defdap`
- If you use conda as your package manager (https://www.anaconda.com/), the prerequisite packages that are available from anaconda can be installed using the command `conda install scipy numpy matplotlib scikit-image scikit-learn pandas networkx jupyter ipython` (Anaconda3-2020.02 has been tested)
- For more information, see the [documentation](https://defdap.readthedocs.io/en/latest/installation.html)

## How to use
- To start the example notebook, use the command `jupyter lab` and click the notebook (.ipynb file)
- Try out the example notebook using the binder link above (Note: change the matplotlib plotting mode to inline, some of the interactive features will not work)
- Here is a video demonstrating the basic functionality of the package: [DefDAP demo](http://www.youtube.com/watch?v=JIbc7F-nFSQ "DefDAP demo")

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
- [networkx](https://networkx.github.io/)
- [pandas](http://pandas.pydata.org)
- [peakutils](https://peakutils.readthedocs.io/en/latest/)
- [matplotlib_scalebar](https://pypi.org/project/matplotlib-scalebar/)
- [IPython](https://ipython.org/)
- [jupyter](https://jupyter.org/)

## License
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
