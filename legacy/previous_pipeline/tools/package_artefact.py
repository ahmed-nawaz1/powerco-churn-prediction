"""Validate and package software, original data and evidence, excluding report PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from powerco.validate import validate


def package(run, destination):
    result = validate(run)
    assert destination.parent.exists()
    files = []
    for name in ['main.py', '.gitignore', 'README.md', 'requirements.txt', 'requirements-report.txt',
                 'client_data.csv', 'price_data.csv', 'cps7008-assessment-1.pdf',
                 'Data Description.docx', 'AI Guidance.docx', 'Assessment Cover Sheet.docx']:
        files.append(ROOT / name)
    for folder in [ROOT / 'powerco', ROOT / 'tools', ROOT / 'tests', ROOT / 'audit', run]:
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in {'.pyc', '.pyo'})
    files = sorted(set(files))
    inventory = {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert not any('WORKING_DRAFT.pdf' in x or 'report_review' in x for x in inventory)
    assert not any('.venv/' in x or '__pycache__' in x for x in inventory)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('PACKAGE_MANIFEST.json', json.dumps(inventory, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        for name, digest in inventory.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        assert 'client_data.csv' in archive.namelist()
        assert 'price_data.csv' in archive.namelist()
        assert 'main.py' in archive.namelist()
    evidence = dict(status='PASS', files=len(inventory), bytes=destination.stat().st_size,
                    zip_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    archive=str(destination.relative_to(ROOT)), report_pdf_excluded=True,
                    checks=['All archive entries hash-matched to source', 'CRC verified',
                            'No virtual environment/caches', 'Original source/data included'],
                    run_validation=result)
    with destination.with_suffix('.validation.json').open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package((ROOT / args.run).resolve(), (ROOT / args.output).resolve())
