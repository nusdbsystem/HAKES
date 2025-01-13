from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="hakesclient",
    version="0.0.1",
    author="guoyu",
    description="hakes client",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="www.comp.nus.edu.sg/~dbsystem/hakes",
    packages=find_packages(),
    install_requires=[
        "tqdm>=4.66.2",
        "numpy>=1.19.5",
    ],
    keywords="vector-database, client",
)
