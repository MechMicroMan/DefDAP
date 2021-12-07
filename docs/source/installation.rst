Installation
===========================

The defdap package is available from the Python Package Index (PyPI) and can be installed by executing the following command: ::

	pip install defdap

The prerequisite packages should be installed automatically by pip, but if you want to manually install them, using the conda package manager for example, then run the following command: ::

	conda install scipy numpy matplotlib scikit-image scikit-learn pandas networkx

If you are doing development work on the scripts, first clone the repository from GitHub. The package can then be installed in editable mode using pip with flag -e to create a "linked" .egg module, which means the module is loaded from the directory at runtime. This avoids having to reinstall every time changes are made. Run the following command from the root of the cloned repository: ::

	pip install -e .