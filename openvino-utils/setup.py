#!/usr/bin/env python
from setuptools import setup

REQUIREMENTS = []
with open('requirements.txt') as requirements_file:
    REQUIREMENTS.extend(requirements_file.readlines())

# dev requirements
with open('requirements-dev.txt') as requirements_file:
    REQUIREMENTS.extend(filter(lambda line: not line.strip().startswith("-r"), requirements_file.readlines()))

setup (
    name='Openvino model wrapper',
    description='An OpenVino model wrapper',
    version='0.1.0',
    packages=['openvino_utils'],
    author='Wajih Ouertani',
    author_email='wajih.ouertani@gmail.com',
    install_requires=REQUIREMENTS,
    scripts=[],
    # package_data={
    # }
)