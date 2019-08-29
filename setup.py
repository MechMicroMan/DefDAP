from setuptools import setup, find_packages

setup(
    name='DefDAP',
    version='0.1',
    description='A package for combined analysis of EBSD and DIC data.',
    author='Michael Atkinson',
    packages=find_packages(),
    install_requires=['scipy',
					  'numpy',
					  'matplotlib',
					  'scikit-image',
					  'pandas',
					  'peakutils',
					  'matplotlib_scalebar',
					  'IPython',
					  'jupyter']
)
