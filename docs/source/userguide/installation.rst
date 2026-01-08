Installation
===========================

For most users, only looking to use DefDAP, the easiest installation method is via the Python Package Index (PyPI).
If you are doing development work on DefDAP, then we reccomend you first clone the repository from GitHub, then install the package in editable mode.

.. tab-set::

	.. tab-item:: PyPi
		
		The latest released version of DefDAP can be installed from PyPI by executing the following command: ::

			pip install defdap

		This will automatically install any required dependencies.

	.. tab-item:: Clone repository
		
		First, clone the repository from GitHub. You can do this multiple ways:

		- (Recomended for new users) Use the GitHub Desktop GUI (https://desktop.github.com/download/).
		- If you are comfortable with the Git version control system, you can clone the repository with the command line.
		- (Not reccomended) Download https://github/com/MechMicroMan/DefDAP as a ZIP file, then extract the contents to a folder on your computer.

		Using the first two methods mean that you can easily pull down updates from the main repository in the future 
		and you can also push any changes you make to your own fork of the repository.

		After cloning the repository, the package can then be installed in editable mode using pip with flag -e.
		This avoids having to reinstall every time changes are made.
		Run the following command from the root of the cloned repository: ::

			pip install -e .

