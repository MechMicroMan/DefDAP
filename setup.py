from setuptools import setup, find_packages


def get_long_description():
    readme_path = 'README.md'
    with open(readme_path, encoding='utf-8') as readme_file:
        contents = readme_file.read()

    return contents


def get_version():
    ver_path = 'defdap/_version.py'
    main_ns = {}
    with open(ver_path) as ver_file:
        exec(ver_file.read(), main_ns)

    return main_ns['__version__']


setup(
    name='DefDAP',
    version=get_version(),
    author='Michael D. Atkinson, Rhys Thomas, João Quinta da Fonseca',
    author_email='michael.atkinson@manchester.ac.uk',
    description='A python library for correlating EBSD and HRDIC data.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    license="Apache 2.0 License",
    keywords='defdap, EBSD, HRDIC, deformation, crystal, correlative analysis',
    project_urls={
        'GitHub': 'https://github.com/MechMicroMan/DefDAP',
        'Documentation': 'https://defdap.readthedocs.io/en/latest'
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Framework :: Matplotlib'
    ],
    packages=find_packages(exclude=['tests']),
    package_data={'defdap': ['slip_systems/*.txt']},
    python_requires='>=3.8',
    install_requires=[
        'scipy>=1.9',
        'numpy',
        'matplotlib>=3.0.0',
        'scikit-image>=0.19',
        'pandas',
        'peakutils',
        'matplotlib_scalebar',
        'networkx',
        'numba',
    ],
    extras_require={
        'testing': ['pytest<8', 'coverage', 'pytest-cov', 'pytest_cases'],
        'docs': [
            'sphinx==5.0.2', 'sphinx_rtd_theme==0.5.0', 
            'sphinx_autodoc_typehints==1.11.1', 'nbsphinx==0.9.3', 
            'ipykernel', 'pandoc', 'ipympl'
        ]
    }
)
