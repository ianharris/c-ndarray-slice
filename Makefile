test:
				pytest

install:
				python setup.py install

uninstall:
				pip uninstall -y slice

clean:
				rm -rf dist build slice.egg-info
				find . -type d -name __pycache__ | xargs rm -r
