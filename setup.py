#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'scikit-learn>=0.21.3',
    'numpy>=1.16.4',
    'jax>=0.1.58',
    #'jaxlib>=0.1.38'
]

setup_requirements = []

test_requirements = []

setup(
    author="Ben Auffarth",
    author_email='auffarth@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Machine learning using jax",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='jax_ml',
    name='jax_ml',
    packages=find_packages(include=['jax_ml', 'jax_ml.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/benman1/jax_ml',
    version='0.1.0',
    zip_safe=False,
)
