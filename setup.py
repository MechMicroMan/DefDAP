from setuptools import setup, find_packages

setup(
    name='DefDAP',
    version='0.92',
    description='A package for combined analysis of EBSD and DIC data.',
    author='Michael D. Atkinson',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'scikit-image',
        'scikit-learn',
        'pandas',
        'peakutils',
        'matplotlib_scalebar',
        'networkx',
        'IPython',
        'jupyter'
    ]
)