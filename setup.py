import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()


def _requires_from_file(filename):
    open_file = os.path.join(here, filename)
    return open(open_file).read().splitlines()


setup(
    name="splade_vocab",
    version="0.0.1",
    author="Hiroki Iida",
    author_email="hiroki.iida@nlp.c.titech.ac.jp",
    description="scripts for splade domain adaptation",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=_requires_from_file("requirements.txt"),
)
