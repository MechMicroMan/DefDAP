from setuptools import setup, find_packages

setup(
    name='DefDAP',
    version='0.92.1',
    description='A python library for correlating EBSD and HRDIC data.',
    author='Michael D. Atkinson',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib>=3.0.0',
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
