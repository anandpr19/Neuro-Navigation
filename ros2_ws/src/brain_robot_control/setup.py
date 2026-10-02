from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'brain_robot_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Neuromorphic Engineering Team',
    maintainer_email='anand@robotics.local',
    description='Brain-Inspired Computing System for Autonomous Robot Control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'neuromorphic_node = brain_robot_control.neuromorphic_node:main',
        ],
    },
)
