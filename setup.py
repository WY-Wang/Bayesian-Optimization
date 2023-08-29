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

    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.3",
        "torch>=2.0.0",
        "gpytorch>=1.10",
        "matplotlib>=3.7.1",
        "pyDOE2",
        "pymoo",
    ],

    packages=setuptools.find_packages(),
)
