#!/usr/bin/env python3
"""Build a self-contained MiShape release with stdlib only, without networking.

Run from any working directory:
    python mishape/scripts/build_release.py
    python mishape/scripts/build_release.py --dry-run

Only this script's explicitly marked output directory/archive can be replaced.
Source files and unrelated output contents are never deleted or modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid
import zipfile


VERSION = '0.1.0'
NAME = f'MiShape-v{VERSION}'
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'mishape'
OUTPUT = ROOT / 'output'
DESTINATION = OUTPUT / NAME
ARCHIVE = OUTPUT / f'{NAME}.zip'
MARKER_NAME = '.mishape-release.json'
MARKER = {
    'schema': 'mishape-managed-release-v1',
    'product': 'MiShape',
    'version': VERSION,
    'directory': NAME,
    'managed_by': 'mishape/scripts/build_release.py',
}
EXCLUDED_DIRS = {
    '.git', '.state', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache',
    '.ruff_cache', 'node_modules', 'output', 'test-results', 'playwright-report',
}
EXAMPLE_NAMES = {'MiShape-Carrera-12-Variants.zip', 'MiShape-Generated-SUV.glb'}
REQUIRED_VERIFICATION = {
    'browser-report.json', 'studio.png', 'comparison.png', 'generator.png',
    'portfolio.png', *EXAMPLE_NAMES,
}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_source(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'源文件缺失或为不支持的符号链接：{path}')
    # A symlinked ancestor could otherwise copy files outside this source tree.
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f'源文件不在项目目录内：{path}')


def skipped(relative):
    parts = relative.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    if relative.name in {'.DS_Store', 'Thumbs.db'} or relative.suffix in {'.pyc', '.pyo'}:
        return True
    if relative.name.endswith('.mishape.json'):
        return True
    # The early inspection image is superseded by final verification screenshots.
    if relative.as_posix() == 'docs/studio-initial.png':
        return True
    # Test replay dumps are large duplicate models, not verification reports.
    if parts and parts[0] in {'docs', 'tests'} and relative.suffix == '.json':
        file = PACKAGE / relative
        if file.is_file() and file.stat().st_size > 2 * 1024 * 1024:
            return True
    return False


def plan_files():
    version_source = PACKAGE / '__init__.py'
    assert_source(version_source)
    found = re.search(r'__version__\s*=\s*[\'\"]([^\'\"]+)', version_source.read_text())
    if not found or found.group(1) != VERSION:
        raise RuntimeError('源码版本与发布脚本版本不一致，请先核对 VERSION。')
    example_sources = {}
    for name in REQUIRED_VERIFICATION:
        source = PACKAGE / 'docs' / 'verification' / name
        if name in EXAMPLE_NAMES and not source.exists():
            source = ROOT / 'examples' / name
        assert_source(source)
        if name in EXAMPLE_NAMES:
            example_sources[name] = source
    for name in ['app.py', 'web/index.html', 'assets/catalog.json', 'README.md',
                 'docs/DELIVERY.md', 'start.sh', 'start.command', 'requirements.txt']:
        assert_source(PACKAGE / name)

    files = {}
    excluded = []
    for source in sorted(PACKAGE.rglob('*')):
        relative = source.relative_to(PACKAGE)
        if skipped(relative):
            if source.is_file():
                excluded.append((Path('mishape') / relative).as_posix())
            continue
        if source.is_symlink():
            raise RuntimeError(f'发布包不接受符号链接：{source}')
        if not source.is_file():
            continue
        assert_source(source)
        if relative.parent == Path('docs/verification') and source.name in EXAMPLE_NAMES:
            target = Path('examples') / source.name
        else:
            target = Path('mishape') / relative
        files[target.as_posix()] = source

    for name, source in example_sources.items():
        files[(Path('examples') / name).as_posix()] = source

    # Only Python files from the legacy package; no web, data, weights or examples.
    dependency = ROOT / 'aeroshape'
    if dependency.is_symlink() or not dependency.is_dir():
        raise RuntimeError('缺少 aeroshape Python 几何依赖目录。')
    for source in sorted(dependency.rglob('*.py')):
        relative = source.relative_to(dependency)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        assert_source(source)
        files[(Path('aeroshape') / relative).as_posix()] = source

    additions = {
        'LICENSE': ROOT / 'LICENSE',
        'aeroshape/LICENSE': ROOT / 'LICENSE',
        'start.sh': PACKAGE / 'start.sh',
        'start.command': PACKAGE / 'start.command',
        'requirements.txt': PACKAGE / 'requirements.txt',
        'tests/test_generation_controls.mjs': ROOT / 'tests/test_generation_controls.mjs',
    }
    for target, source in additions.items():
        assert_source(source)
        files[target] = source
    if not any(name.startswith('mishape/tests/test_') and name.endswith('.py') for name in files):
        raise RuntimeError('没有找到 MiShape 的 Python 测试。')
    return files, excluded


def check_managed_directory(path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f'拒绝替换非目录或符号链接：{path}')
    marker_path = path / MARKER_NAME
    try:
        if marker_path.is_symlink():
            raise ValueError('symlink marker')
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'目标已存在且未标明由本脚本管理，拒绝删除：{path}') from exc
    if marker != MARKER:
        raise RuntimeError(f'目标的管理标识不匹配，拒绝删除：{path}')


def check_managed_archive(path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'拒绝替换非文件或符号链接：{path}')
    try:
        with zipfile.ZipFile(path) as archive:
            marker = json.loads(archive.read(f'{NAME}/{MARKER_NAME}'))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f'ZIP 已存在且未确认由本脚本管理，拒绝覆盖：{path}') from exc
    if marker != MARKER:
        raise RuntimeError(f'ZIP 管理标识不匹配，拒绝覆盖：{path}')


@contextmanager
def build_lock():
    path = OUTPUT / f'.{NAME}.build.lock'
    token = json_text({'product': 'MiShape', 'pid': os.getpid(), 'token': uuid.uuid4().hex})
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'已有构建锁：{path}；请确认没有其他构建运行后再处理该锁文件。') from exc
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(token)
        yield
    finally:
        if path.is_file() and not path.is_symlink() and path.read_text() == token:
            path.unlink()


def write_release_readme(stage):
    (stage / 'README.md').write_text('''# MiShape 0.1 — polygon 车辆设计工作台

本包包含独立 MiShape 软件、两辆 Porsche 素材、参数化生成器、使用文档及验证记录。旧 AeroShape 仅作为 Python 几何依赖提供，启动器运行 MiShape。

## 启动

macOS 双击本目录 `start.command`，或在终端运行：

```sh
bash start.sh
```

需要 Python 3.11+；首次运行可自动建立虚拟环境并安装依赖。已有依赖后可在本机运行。服务地址为 [http://127.0.0.1:8010](http://127.0.0.1:8010)，按终端 `Ctrl+C` 停止。`bash start.sh --check` 只检查环境。

## 内容

- [快速使用说明](mishape/README.md)
- [完整操作与技术说明](mishape/docs/DELIVERY.md)
- [素材数据说明](mishape/assets/README.md)
- [浏览器验证报告](mishape/docs/verification/browser-report.json)；同目录包含最终界面截图和其他验证记录。
- `examples/MiShape-Carrera-12-Variants.zip`：实际 12 变体 OBJ、配方与质量记录。
- `examples/MiShape-Generated-SUV.glb`：实际生成的 SUV 几何示例。
- `mishape/tests/` 与 `tests/test_generation_controls.mjs`：Python 和生成参数联动测试。
- `SHA256SUMS`：包内文件的 SHA-256；`release-info.json`：发布内容说明。

塑形和生成用于概念设计；当前版本没有制造、公差、完整自交或 CFD 认证。内置车尺寸为待校准估计，界面显示近似材质，原 UV/纹理副本独立保存。软件许可见 LICENSE；用户提供的车辆素材不由该软件许可自动授予再分发权。

开发测试需另外安装 pytest、httpx；Node 测试可通过 `PYTHON` 环境变量指定虚拟环境中的 Python。此包不包含旧界面、CFD 权重、原工作区缓存或测试用重型工程副本。
''', encoding='utf-8')


def write_checksums(stage):
    hashes = {}
    for path in sorted(stage.rglob('*')):
        if path.is_file() and path != stage / 'SHA256SUMS':
            hashes[path.relative_to(stage).as_posix()] = sha256(path)
    (stage / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    return hashes


def build_zip(stage, target):
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for source in sorted(stage.rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(f'{NAME}/{relative}', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ((stat.S_IFREG | stat.S_IMODE(source.stat().st_mode)) << 16)
            # Streaming avoids holding Blend/GLB/model archives in memory.
            with source.open('rb') as input_file, archive.open(info, 'w', force_zip64=True) as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    with zipfile.ZipFile(target) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError(f'发布 ZIP CRC 检查失败：{broken}')


def verify_stage(stage, hashes):
    for relative, expected in hashes.items():
        if sha256(stage / relative) != expected:
            raise RuntimeError(f'SHA-256 检查失败：{relative}')
    for name in EXAMPLE_NAMES:
        if name.endswith('.zip'):
            with zipfile.ZipFile(stage / 'examples' / name) as archive:
                broken = archive.testzip()
                if broken:
                    raise RuntimeError(f'示例 ZIP CRC 检查失败：{broken}')
    for source in stage.rglob('*'):
        if any(part in EXCLUDED_DIRS for part in source.relative_to(stage).parts):
            raise RuntimeError(f'发布包意外包含缓存或状态目录：{source}')


def commit(stage, staged_zip, temporary):
    """Commit both artifacts, restoring previous managed output on any failure."""
    check_managed_directory(DESTINATION)
    check_managed_archive(ARCHIVE)
    old_directory = temporary / 'previous-directory'
    old_archive = temporary / 'previous-archive.zip'
    installed_directory = installed_archive = False
    try:
        if DESTINATION.exists():
            os.replace(DESTINATION, old_directory)
        if ARCHIVE.exists():
            os.replace(ARCHIVE, old_archive)
        os.replace(stage, DESTINATION)
        installed_directory = True
        os.replace(staged_zip, ARCHIVE)
        installed_archive = True
    except BaseException:
        if installed_archive:
            check_managed_archive(ARCHIVE)
            ARCHIVE.unlink()
        if installed_directory:
            check_managed_directory(DESTINATION)
            shutil.rmtree(DESTINATION)
        if old_directory.exists():
            os.replace(old_directory, DESTINATION)
        if old_archive.exists():
            os.replace(old_archive, ARCHIVE)
        raise
    # TemporaryDirectory removes only the previous, positively identified outputs.


def main(argv=None):
    parser = argparse.ArgumentParser(description='构建 MiShape 本地独立发布包（仅标准库）')
    parser.add_argument('--dry-run', action='store_true', help='只检查来源和目标，不创建/删除/打包文件')
    args = parser.parse_args(argv)
    if OUTPUT.is_symlink():
        raise RuntimeError('output 为符号链接；为避免影响其他目录，本脚本拒绝继续。')
    files, excluded = plan_files()
    check_managed_directory(DESTINATION)
    check_managed_archive(ARCHIVE)
    source_bytes = sum(path.stat().st_size for path in files.values())
    if args.dry_run:
        print(json_text({'mode': 'dry-run', 'files': len(files), 'source_bytes': source_bytes,
                         'directory': str(DESTINATION), 'archive': str(ARCHIVE),
                         'excluded_files': excluded}))
        return

    OUTPUT.mkdir(exist_ok=True)
    with build_lock():
        with tempfile.TemporaryDirectory(prefix=f'.{NAME}-build-', dir=OUTPUT) as scratch:
            temporary = Path(scratch)
            stage = temporary / NAME
            stage.mkdir()
            for target, source in files.items():
                output = stage / target
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)
                output.chmod(0o755 if output.suffix in {'.sh', '.command'} else 0o644)
            (stage / MARKER_NAME).write_text(json_text(MARKER))
            write_release_readme(stage)
            (stage / 'release-info.json').write_text(json_text({
                'product': 'MiShape', 'version': VERSION, 'schema': 'mishape-release-v1',
                'source_file_count': len(files), 'source_bytes': source_bytes,
                'included': ['MiShape source/web/assets/docs/tests', 'aeroshape Python modules + LICENSE',
                             'final verification reports/screenshots', '12-variant ZIP and generated SUV GLB'],
                'excluded': ['runtime .state', 'virtual environments and caches', 'legacy web/weights/samples',
                             'large replay project JSON', *excluded],
                'examples_relocated_from': 'mishape/docs/verification',
                'network_access': False, 'zip_crc_check': 'required_before_commit',
                'checksums': 'SHA256SUMS covers every release file except itself',
            }))
            hashes = write_checksums(stage)
            verify_stage(stage, hashes)
            staged_zip = temporary / f'{NAME}.zip'
            build_zip(stage, staged_zip)
            archive_hash = sha256(staged_zip)
            archive_bytes = staged_zip.stat().st_size
            commit(stage, staged_zip, temporary)
    print(json_text({'status': 'complete', 'directory': str(DESTINATION), 'archive': str(ARCHIVE),
                     'files': len(hashes) + 1, 'archive_bytes': archive_bytes,
                     'archive_sha256': archive_hash, 'checksums': 'verified', 'zip_crc': 'passed'}))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f'MiShape 打包失败：{error}') from error
