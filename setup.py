from setuptools import setup, find_packages

setup(
	name="tuningdeap",
	version="0.0.3",
	description="A package to tune Machine Learning hyperparameters with config files",
	long_description=None,
	url= "https://github.com/orionw/tuningDEAP",
	author="Brandon Schoenfeld and Orion Weller",
	author_email="bsjchoenfeld@byu.edu",
	license="MIT",
	classifiers=[
		"Programming Language :: Python :: 3.5",
	],
    download_url="https://github.com/orionw/tuningDEAP/archive/v0.0.3.tar.gz",
	keywords="tuning machinelearning genetic hyperparameters",
	packages=find_packages(),
	install_requires=["deap", "numpy", "pandas"],
	python_requires="~=3.5",
)
