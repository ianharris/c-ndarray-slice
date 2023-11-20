# C ndarray slice

A demonstration of using NumPy's C-API to create a function to slice an ndarray along the 0-axis.

A Makefile is used for:

* installing / uninstalling,
* cleaning build and \_\_pycache\_\_ files,
* running tests.

## Requirements

Requires requires Python packages numpy and pytest to be installed.

## Testing

Install the package (ideally in a virtual environment) using `make install` and then run
the tests using `make test`.

## Notes on Previous Versions

A previous version (specifically, commit 9ebf0e5) used an inefficient copy mechanism.
And worse had a system level `malloc` call. That version, shouldn't be used.
