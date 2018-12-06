from setuptools import setup, find_packages
from codecs import open
from os import path

# get the version
exec(open('version.py').read())

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='sweepable',
    version=__version__,
    description='A framework for numerical experiments.',
    long_description=long_description,
    packages=find_packages(),
    author='Benjamin Margolis',
    author_email='ben@sixpearls.com',
    url='https://github.com/simupy/sweepable',
    license="BSD 2-clause \"Simplified\" License",
    python_requires='>=3',
    install_requires=['peewee>=3.7.1',],

    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
