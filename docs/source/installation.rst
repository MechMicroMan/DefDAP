DefDAP Installation
===========================

To use DefDAP it must be installed locally as a python module. The package is available from the Python Package Index (PyPI) and can be installed by executing the command.

From the root DefDAP folder execute the command. ::

	pip install defdap

If you use conda as your package manager, the prerequisite packages that are available from anaconda can be installed using the command. ::

	conda install scipy numpy matplotlib scikit-image scikit-learn pandas networkx jupyter ipython

If you are doing development work on the scripts, first clone the repository from GitHub. The package can then be installed in editable using pip with flag -e to create a "linked" .egg module, which means the module is loaded from the directory at runtime. This avoids having to reinstall every time changes are made. ::

	pip install -e .