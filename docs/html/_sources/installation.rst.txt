DefDAP Installation
===========================

To use DefDAP it must be installed locally as a python module. From the root DefDAP folder execute the command. ::

	pip install .

If you are doing development work on the scripts, you can use the -e flag to create a "linked" .egg module which means the module is loaded from the directory at runtime. This avoids having to reinstall every time changes are made. ::

	pip install -e .