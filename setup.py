"""
package setup
"""

import io
import os
import re
import codecs

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(HERE, *parts), "r").read()


def read_requirements(*parts):
    """
    Return requirements from parts.
    Given a requirements.txt (or similar style file),
    returns a list of requirements.
    Assumes anything after a single '#' on a line is a comment, and ignores
    empty lines.
    :param parts: list of filenames which contain the installation "parts",
        i.e. submodule-specific installation requirements
    :returns: A compiled list of requirements.
    """
    requirements = []
    for line in read(*parts).splitlines():
        new_line = re.sub(  # noqa: PD005
            r"(\s*)?#.*$",  # the space immediately before the
            # hash mark, the hash mark, and
            # anything that follows it
            "",  # replace with a blank string
            line,
        )
        new_line = re.sub(  # noqa: PD005
            r"-r.*$",  # link to another requirement file
            "",  # replace with a blank string
            new_line,
        )
        new_line = re.sub(  # noqa: PD005
            r"-e \..*$",  # link to editable install
            "",  # replace with a blank string
            new_line,
        )
        # print(line, "-->", new_line)
        if new_line:  # i.e. we have a non-zero-length string
            requirements.append(new_line)
    return requirements


INSTALL_REQUIRES = read_requirements(".requirements/base.in")
EXTRA_REQUIRES = {
    "dev": read_requirements(".requirements/dev.in"),
    "docs": read_requirements(".requirements/docs.in"),
    "cpu": read_requirements(".requirements/cpu.in"),
    "cu116": read_requirements(".requirements/cu116.in"),
    "cu117": read_requirements(".requirements/cu117.in"),
}
# Add all requires
all_requires = []
for k, v in EXTRA_REQUIRES.items():
    if k not in ["cu116", "cu117"]:
        all_requires.extend(v)
EXTRA_REQUIRES["all"] = set(all_requires)


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("gauche", "__init__.py")
readme = open("README.md").read()
packages = find_packages(".", exclude=["tests"])


setup(
    name="gauche",
    version=version,
    description="Gaussian Process Library for Molecules, Proteins and General Chemistry in PyTorch.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="machine-learning gaussian-processes kernels pytorch chemistry biology protein ligand",
    packages=packages,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires=">=3.8",
    platforms="any",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
)
