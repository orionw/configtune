from setuptools import setup, find_packages

setup(
	name="tuningdeap",
	version="0.0.1",
	description="A package to tune Machine Learning hyperparameters with config files",
	long_description=None,
	url= "https://github.com/orionw/tuningDEAP",
	author="Brandon Schoenfeld and Orion Weller",
	author_email="bsjchoenfeld@byu.edu",
	license="MIT",
	classifiers=[
		"Programming Language :: Python :: 3.5",
	],
	keywords="tuning macinelearning genetic hyperparameters",
	packages=find_packages(),
	install_requires=["deap", "numpy"],
	python_requires="~=3.5",
)