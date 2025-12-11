# pylie: Lie groups in Python

This package implements common Lie groups in python.
The idea is to have Lie groups interact as you would expect: multiplication operators are defined for group products as well as default group actions on vectors.
There are also some analysis and plotting tools.

## Tests

To run the tests, use the following commands in the main directory.

```sh
python3 -m unittest ./tests/test_LieGroups.py
python3 -m unittest ./tests/test_analysis.py
```

## Installation

This project is available on PyPI as a package, but you can also install it locally by running the following from the main directory.

```sh
python3 -m pip install -e .
```
