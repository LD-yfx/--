from setuptools import setup
from glob import glob

setup(name='quality_navigation_demo', version='0.1.0', packages=['quality_navigation_demo'],
      data_files=[('share/ament_index/resource_index/packages',['resource/quality_navigation_demo']),
                  ('share/quality_navigation_demo',['package.xml','README.md']),
                  ('share/quality_navigation_demo/launch',glob('launch/*.py')),
                  ('share/quality_navigation_demo/config',glob('config/*'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='SITP team', maintainer_email='sitp@example.invalid', license='Apache-2.0',
      description='Synthetic closed-loop Nav2 mechanism tests',
      entry_points={'console_scripts':[
          'kinematic_sim = quality_navigation_demo.simulator:main',
          'integration_runner = quality_navigation_demo.runner:main']})
