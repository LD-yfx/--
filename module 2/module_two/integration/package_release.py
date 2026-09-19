#!/usr/bin/env python3
"""Build and verify the source release without changing the working tree."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main():
    module = Path(__file__).resolve().parents[1]
    output = module.parent / 'SITP_模块二_整合修正版_20260919.zip'
    excluded_dirs = {'build', 'install', 'log', '__pycache__', '.pytest_cache',
                     '.git', 'debug', 'bag', 'bags'}
    excluded_suffixes = {'.pyc', '.pyo', '.db3', '.mcap', '.zstd', '.zst', '.log'}
    entries = {}
    for folder, dirs, names in os.walk(module):
        dirs[:] = sorted(d for d in dirs if d not in excluded_dirs and
                         not d.endswith('.egg-info') and
                         d not in {'runner_smoke_01', 'scatter_fixture_20260919'})
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix in excluded_suffixes:
                continue
            if path.is_symlink():
                raise RuntimeError('Unexpected source symlink: ' + str(path))
            archive_name = 'module_two/' + path.relative_to(module).as_posix()
            content = path.read_bytes()
            if path.suffix == '.py':
                ast.parse(content, filename=archive_name)
            if path.suffix == '.sh' and b'\r\n' in content:
                raise RuntimeError('Shell script must use LF: ' + archive_name)
            entries[archive_name] = content
    packages = sorted(ET.fromstring(content).findtext('name')
                      for name, content in entries.items()
                      if name.startswith('module_two/ros2_ws/src/') and
                      name.count('/') == 4 and name.endswith('/package.xml'))
    expected = sorted(['localization_quality', 'quality_navigation_msgs',
                       'quality_aware_navigation', 'quality_nav2_layer',
                       'quality_navigation_demo', 'quality_localization_benchmark',
                       'quality_localization_audit'])
    assert packages == expected, packages
    result = json.loads(entries['module_two/results/research/runner_smoke_02/result.json'])
    assert all(result[k] is True for k in ('success', 'valid_evidence', 'pose_metrics_valid'))
    assert result['truth_coverage'] == result['estimate_coverage'] == 1.0
    assert result['collision_count'] == 0 and len(result['legs']) == 2
    entries['解压后先读.md'] = (
        '# 模块二整合修正版\n\n'
        '请先阅读 [调整内容、测试结果及运行步骤](module_two/docs/release_notes.md)。\n\n'
        '包内包含七个 ROS2 包、启动配置、测试和报告。Ubuntu 22.04 / ROS2 Humble '
        '下重新构建后运行；不包含编译缓存和大型 rosbag。\n\n'
        '本次独立定位往返测试通过；正式质量收益对照实验和新增动态 ROS 集成尚未完成。\n'
    ).encode('utf-8')
    manifest = {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                for name, content in sorted(entries.items())}
    entries['MANIFEST_SHA256.json'] = (json.dumps(
        {'format_version': 1, 'packages': packages, 'files': manifest},
        indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(entries.items()):
            info = ZipInfo(name, (2026, 9, 19, 0, 0, 0))
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            mode = 0o755 if name.endswith('.sh') else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, content)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(entries)
        for name, expected_file in manifest.items():
            content = archive.read(name)
            assert len(content) == expected_file['bytes']
            assert hashlib.sha256(content).hexdigest() == expected_file['sha256']
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(digest + '  ' + output.name + '\n', encoding='utf-8')
    print(json.dumps({'zip': str(output), 'files': len(entries), 'packages': packages,
                      'bytes': output.stat().st_size, 'sha256': digest,
                      'crc_and_manifest_verified': True}, ensure_ascii=True))


if __name__ == '__main__':
    main()
