from setuptools import setup, find_packages


def get_long_description():
    readme_file = 'README.md'
    with open(readme_file, encoding='utf-8') as handle:
        contents = handle.read()

    return contents


setup(
    name='DefDAP',
    version='0.92.3',
    author='Michael D. Atkinson, Rhys Thomas, João Quinta da Fonseca',
    author_email='michael.atkinson@manchester.ac.uk',
    description='A python library for correlating EBSD and HRDIC data.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    license="Apache 2.0 License",
    keywords='defdap, EBSD, HRDIC, deformation, crystal, correlative analysis',
    project_urls={
        'GitHub': 'https://github.com/MechMicroMan/DefDAP',
        'Documentation': 'http://mechmicroman.github.io/DefDAP'
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Framework :: Matplotlib'
    ],
    packages=find_packages(exclude=['tests']),
    package_data={'defdap': ['slip_systems/*.txt']},
    python_requires='>=3.5',
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
    ],
    extras_require={
        'testing': ['pytest', 'coverage', 'pytest-cov', 'pytest_cases']
    }

)
