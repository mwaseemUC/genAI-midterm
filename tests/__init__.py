"""Test package.

This file is required, not incidental. With pytest's default "prepend" import
mode, the directory prepended to sys.path is the highest ancestor of the test
file still containing an __init__.py. Without this file that would be tests/,
so `import rag` and `import tests.fakes` would both fail. With it, the
repository root is prepended and both resolve.
"""
