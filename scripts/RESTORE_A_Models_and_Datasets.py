"""Verify and restore the one PLD A models-and-datasets ZIP. Standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

MANIFEST = 'A_MODELS_DATASETS_MANIFEST_20261017.json'
ARCHIVE = 'PLD_A_Models_and_Datasets_20261017.zip'
SCHEMA = 'pld-a-models-datasets-distribution-v1'
BLOCK = 4 * 1024 * 1024


def publish_without_replacement(source, destination):
    if os.name == 'nt':
        # Windows rename fails atomically if the destination already exists.
        os.rename(source, destination)
    else:
        # A link in the same directory publishes atomically and fails on EEXIST.
        # Unsupported filesystems fail safely; never fall back to replacing rename.
        os.link(source, destination)
        source.unlink()


def load_manifest(directory):
    data = json.loads((directory / MANIFEST).read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or data.get('archive', {}).get('name') != ARCHIVE:
        raise ValueError('Unsupported bundle manifest or archive name')
    archive = data['archive']
    parts = data.get('parts')
    if not isinstance(parts, list) or not 1 <= len(parts) <= 100:
        raise ValueError('Invalid part list')
    for index, part in enumerate(parts, 1):
        if part.get('name') != ARCHIVE + '.%03d' % index:
            raise ValueError('Parts must be consecutive and stay in this folder')
        if type(part.get('bytes')) is not int or not 0 < part['bytes'] < 2 * 1024**3:
            raise ValueError('Invalid part size')
        if not isinstance(part.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', part['sha256']):
            raise ValueError('Invalid part hash')
    if type(archive.get('bytes')) is not int or archive['bytes'] != sum(p['bytes'] for p in parts):
        raise ValueError('Archive size does not match the parts')
    if not isinstance(archive.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', archive['sha256']):
        raise ValueError('Invalid archive hash')
    return data


def restore(directory, verify_only=False, output=None):
    directory = Path(directory).resolve()
    data = load_manifest(directory)
    archive, parts = data['archive'], data['parts']
    for part in parts:
        source = directory / part['name']
        if not source.is_file() or source.stat().st_size != part['bytes']:
            raise ValueError('Missing or incomplete part: ' + part['name'])
    output = Path(output).resolve() if output else directory / ARCHIVE
    if not verify_only and output.exists():
        raise FileExistsError('Output already exists; use --verify-only to check parts: ' + str(output))
    if not verify_only and shutil.disk_usage(output.parent).free < archive['bytes'] + 16*1024*1024:
        raise OSError('Not enough free space for the restored ZIP')
    temporary = None
    target = None
    combined = hashlib.sha256()
    try:
        if not verify_only:
            target = tempfile.NamedTemporaryFile(prefix=ARCHIVE+'.', suffix='.assembling', dir=output.parent, delete=False)
            temporary = Path(target.name)
        for index, part in enumerate(parts, 1):
            digest = hashlib.sha256()
            with (directory / part['name']).open('rb') as source:
                while True:
                    block = source.read(BLOCK)
                    if not block:
                        break
                    digest.update(block)
                    combined.update(block)
                    if target is not None:
                        target.write(block)
            if digest.hexdigest() != part['sha256']:
                raise ValueError('SHA256 mismatch: ' + part['name'])
            print('Verified part %d/%d: %s' % (index, len(parts), part['name']), flush=True)
        if combined.hexdigest() != archive['sha256']:
            raise ValueError('Combined ZIP SHA256 mismatch')
        if target is not None:
            target.flush()
            os.fsync(target.fileno())
            target.close()
            target = None
            if not zipfile.is_zipfile(temporary):
                raise ValueError('Restored file is not a valid ZIP')
            if output.exists():
                raise FileExistsError('Output appeared during restoration; it was not overwritten')
            publish_without_replacement(temporary, output)
            temporary = None
        print('All hashes match. ' + ('Verification complete.' if verify_only else 'ZIP ready: '+str(output)), flush=True)
        return output
    finally:
        if target is not None:
            target.close()
        if temporary is not None and temporary.is_file():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description='还原 A 组模型与数据集总 ZIP；不会解压数据或运行模型')
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--verify-only', action='store_true', help='只校验分卷与整包哈希，不写入 ZIP')
    parser.add_argument('--output', type=Path, help='新 ZIP 的输出路径；不覆盖已有文件')
    args = parser.parse_args()
    try:
        restore(args.directory, args.verify_only, args.output)
    except Exception as exc:
        print('FAILED: '+str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
