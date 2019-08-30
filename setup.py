from setuptools import setup, find_packages

setup(
	name="configtune",
	version="0.0.8",
	description="A package to tune parameters with config files: designed with machine learning hyperparameter tuning in mind.",
	long_description=None,
	url= "https://github.com/orionw/configtune",
	author="Orion Weller and Brandon Schoenfeld",
	author_email="orionw@byu.edu",
	license="MIT",
	classifiers=[
		"Programming Language :: Python :: 3.5",
	],
    download_url="https://github.com/orionw/configtune/archive/v0.0.8.tar.gz",
	keywords="tuning machinelearning genetic hyperparameters bayesian optimization",
	packages=find_packages(),
	install_requires=["deap", "numpy", "pandas", "scikit-optimize"],
	python_requires="~=3.5",
)
