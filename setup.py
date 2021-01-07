"""
package setup
"""

import io
import os
import re
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("gprotorch", "__init__.py")
readme = open("README.md").read()
packages = find_packages(".", exclude=["tests"])
install_requires = open("requirements.txt").read().splitlines()


setup(
    name="gprotorch",
    version=version,
    description="Gaussian Process Library for Molecules, Proteins and General Chemistry in PyTorch.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="machine-learning gaussian-processes kernels pytorch chemistry biology protein ligand",
    url="https://github.com/leojklarner/GProTorch",
    packages=packages,
    install_requires=install_requires,
    python_requires=">=3.8",
)
