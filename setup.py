"""
Packaging setup script.
This file is required by setuptools.
All configuration data is in the setup.cfg file.
"""

from setuptools import setup, find_packages

# setup(
#   packages = [''],
#   version='0.0.1',
#   package_dir = {'': 'src'},
# )

setup(
    # ...
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        include=['lfiax*'],  # ['*'] by default
        exclude=['lfiax.tests'],  # empty by default
    ),
)