# pylie: Lie groups in Python

This package implements common Lie groups in python.
The idea is to have Lie groups interact as you would expect: multiplication operators are defined for group products as well as default group actions on vectors.
There are also some analysis and plotting tools.

## Installation

Run setup and install:

`python3 -m build && python3 -m pip install dist/pylie-0.1.0.tar.gz`

### Possible Issues

If python says there is no module called 'build', then you probably need to install build:

`python3 -m pip install build`

If the installation says it is installing 'UNKNOWN-0.0.0' then you need to upgrade pip:

`pip install --upgrade pip`
