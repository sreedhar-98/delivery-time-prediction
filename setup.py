from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(path:str)->List[str]:
    requirements=[]
    with open(path) as f:
        requirements=f.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name="House-Price_prediction",
    version='0.0.1',
    author='Sreedhar',
    author_email='sridhardronamraju1@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
