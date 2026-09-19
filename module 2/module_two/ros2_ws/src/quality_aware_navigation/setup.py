from glob import glob
from setuptools import find_packages, setup

setup(
    name='quality_aware_navigation', version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/quality_aware_navigation']),
        ('share/quality_aware_navigation', ['package.xml']),
        ('share/quality_aware_navigation/config', glob('config/*.yaml')),
        ('share/quality_aware_navigation/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'], zip_safe=True,
    maintainer='SITP team', maintainer_email='maintainer@example.invalid',
    description='Quality-aware navigation cost model and ROS adapter', license='Apache-2.0',
    entry_points={'console_scripts': [
        'quality_cost_node = quality_aware_navigation.ros_node:main',
        'quality_to_cost = quality_aware_navigation.cli:main',
    ]},
)
