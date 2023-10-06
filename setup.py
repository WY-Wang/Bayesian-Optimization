import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BayesianOpt",
    version="0.0.0",

    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Wenyu Wang",
    author_email="wenyu_wang@u.nus.edu",

    python_requires=">=3.9",
    install_requires=[
        "torch~=2.0.0",
        "gpytorch~=1.11",
        "botorch~=0.9.0",
        "numpy",
        "matplotlib",
        "pyDOE2",
        "pymoo",
    ],

    packages=setuptools.find_packages(),
)
