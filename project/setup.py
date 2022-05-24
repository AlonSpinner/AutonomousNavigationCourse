from setuptools import setup
from setuptools import find_packages

setup(
    name = "cbMDP",
    version = "1.0.0",
    description = "implementation of 'Planning in the Continous Domain: \
		a Generalized Belief Space Approach for Autonomous Navigation in Unknown Enviorments'\
		by Vadim Indelamn, Luca Carlone and Frank Dellaret",
    author = "Alon and Dan",
    packages = find_packages(exclude = ('tests*')),
    )
