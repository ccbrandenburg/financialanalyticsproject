#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='iembdfa',
    version='1.3',
    description="pip install iembdafa \n Included in this module are 5 libraries that will help you during your data science adventures and help you save some of that valuable time you would rather spend on modelling than on data cleaning.",
    long_description=readme + '\n\n' + history,
    author="TeamD, Christopher Brandenburg",
    author_email='cbrandenburg@student.ie.edu',
    url='https://github.com/ccbrandenburg/financialanalyticsproject/0.1',
    packages=[
        'iembdfa',
    ],
    package_dir={'iembdfa':
                 'iembdfa'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='iembdfa',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
