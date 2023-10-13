from __future__ import annotations

import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the rrls version."""
    path = CWD / "rrls" / "__version__.py"
    about = {}  # type: ignore
    with open(path) as f:
        exec(f.read(), about)
    return about["__version__"]


setup(
    name="rrls",
    version=get_version(),
    long_description=open("README.md").read(),
)
