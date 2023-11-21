import sys
import os
from setuptools import setup, find_packages

readme = open('README.rst').read()

requires = ['numpy>=1.18.1',
            'lenstronomy',
            'scipy>=1.4.1',
            'matplotlib',
            'fast_histogram',
            'tqdm',
            'pycbc',
            'jax==0.2.17',
            'jaxlib==0.1.65'
           ]

setup(
    name='wolensing',
    version='0.0',
    description='Computing the wave optics lensing',
    long_description=readme,
    author='Simon Man-Chun Yeung, Mark Ho-Yeuk Cheung',
    author_email='yeungm@uwm.edu',
    url='',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requires,
    license='MIT',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ]
)
