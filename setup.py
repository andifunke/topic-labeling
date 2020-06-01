#!/usr/bin/env python3
#
# Copyright (c) 2019-present, Ella Media GmbH
# All rights reserved.
#
"""
The recommended way to install the topic-labeling package in an existing
environment is:

```bash
python setup.py develop
```
"""

import subprocess
from pathlib import Path

import setuptools


# --- constants ---

URL = "https://github.com/andifunke/topic-labeling"
README = "README.md"
PACKAGE = "topic-labeling"
PACKAGE_DIR = Path('./src') / PACKAGE.replace('-', '_')
DEFAULT_SPACY_MODEL = 'de'


# --- functions ---

def install_spacy_model(model=DEFAULT_SPACY_MODEL):
    try:
        import de_core_news_md
    except ImportError:
        subprocess.run(['python', '-m', 'spacy', 'download', model])


def read_version():
    print('inferring version')
    try:
        with open(PACKAGE_DIR.resolve() / '__init__.py') as fp:
            for line in fp.readlines():
                if line.startswith('__version__'):
                    version = line.lstrip("__version__ = '").rstrip("'\n")
                    print('version:', version)
                    return version
    except FileNotFoundError as e:
        print('info:', e)
        return None


def read_readme():
    try:
        with open(README, 'r') as fp:
            print('reading', README)
            readme = fp.read()
    except OSError:
        print("README.md not found.")
        readme = ""
    return readme


# --- main ---

setuptools.setup(
    name=PACKAGE,
    version=read_version(),
    author="Andreas Funke",
    author_email="andreas.funke@uni-duesseldorf.de",
    description="The topic-labeling python package",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={'Source': URL},
    python_requires=">=3.6",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
    platforms=['any'],
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={PACKAGE: ['*.cfg']},
    exclude_package_data={'': ['setup.cfg']},
    entry_points={
        'console_scripts': [
        ],
    },
)

# --- post-install ---

install_spacy_model()
