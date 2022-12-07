"""
Packaging setup script.
This file is required by setuptools.
All configuration data is in the setup.cfg file.
"""

from setuptools import setup, find_packages

# setup(name='sbideoman', version='0.0.1', packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),)
setup(
  packages = ['lfiax'],
  package_dir = {'': 'src'},
)

# name='sbidoeman',
#      version='0.1.0a2',
#      packages=
# find_packages(where='src', exclude=["tests", "*.tests", "*.tests.*", "tests.*"])
#  include=["src*"], 
# ['src.minebed', 'src.promisys', 'src.minebed.static']
# ['src']
