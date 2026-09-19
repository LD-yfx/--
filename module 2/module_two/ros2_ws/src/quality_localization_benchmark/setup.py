from pathlib import Path
from setuptools import setup

name='quality_localization_benchmark'
data=[('share/ament_index/resource_index/packages',['resource/'+name]),
      ('share/'+name,['package.xml'])]
for directory in ('config','launch','robot','worlds'):
    for path in Path(directory).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            data.append(('share/'+name+'/'+str(path.parent),[str(path)]))
setup(name=name, version='0.2.0', packages=[name], data_files=data,
      install_requires=['setuptools'], tests_require=['pytest'], zip_safe=False,
      maintainer='SITP team', maintainer_email='sitp@example.invalid', license='Apache-2.0',
      description='Independent localization and sensor-derived quality navigation benchmark',
      entry_points={'console_scripts':[
          'benchmark_worlds = quality_localization_benchmark.worlds:main',
          'sensor_bridge = quality_localization_benchmark.sensor_bridge:main',
          'localization_adapter = quality_localization_benchmark.localization_adapter:main',
          'benchmark_runner = quality_localization_benchmark.runner:main',
          'benchmark_campaign = quality_localization_benchmark.campaign:main',
          'benchmark_plot = quality_localization_benchmark.render_trial:main',
      ]})
