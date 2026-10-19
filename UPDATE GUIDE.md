# Updating on Pypi

The following instructions are mostly based on (https://packaging.python.org/en/latest/tutorials/packaging-projects/) and are copied here to remind me how to update pylie.

1. To update the code, first use `bumpver update --patch/minor/major` to update the version information after having committed the changes.
2. To build the code, us `python3 -m build`. Make sure to remove old files in dist/ before uploading.
3. To upload to Pypi, use `python3 -m twine upload --repository pylieg dist/*`. This will require API tokens.
