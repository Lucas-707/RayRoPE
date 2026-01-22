from setuptools import find_packages, setup

setup(
    name="prope-nvs",
    version="0.0.0",
    packages=find_packages(include=["prope*", "nvs*"]),
    package_dir={"": "."},
    license="MIT",
)
