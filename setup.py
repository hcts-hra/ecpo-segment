# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="ecpo_segment",
    version="0.0.1",
    description="Segmenting Early Chinese Periodicals",
    url="https://github.com/exc-asia-and-europe/ecpo-segment",
    author="Simon Will",
    author_email='simon.will@altechnologies.de',
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "dh_segment",
        "numpy",
        "Pillow",
        "requests",
    ],
    tests_require=[
        "numpy",
        "Pillow",
        "pytest",
        "responses",
    ],
    setup_requires=[
        "pytest-runner"
    ],
)
