from setuptools import setup, find_packages

setup(
    name="crystallm",
    version="1.0",
    packages=find_packages(include=["crystallm"]),
    package_data={"crystallm": ["*.txt"]},
)
