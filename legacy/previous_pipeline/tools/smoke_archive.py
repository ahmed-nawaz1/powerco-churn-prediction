"""Verify the packaged artefact from a new extracted working directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def smoke(archive, run):
    staging = ROOT / 'tmp'
    staging.mkdir(exist_ok=True)
    extracted = Path(tempfile.mkdtemp(prefix='archive_smoke_', dir=staging)).resolve()
    assert extracted.is_relative_to(staging.resolve())
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (extracted / entry.filename).resolve()
            assert target.is_relative_to(extracted), 'Unsafe archive path'
        z.extractall(extracted)
    commands = [[sys.executable, '-m', 'powerco.validate', run],
                [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    checks = []
    for command in commands:
        result = subprocess.run(command, cwd=extracted, capture_output=True, text=True)
        checks.append(dict(command=command, returncode=result.returncode,
                           stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, json.dumps(checks, indent=2)
    evidence = dict(status='PASS', extracted_to=str(extracted.relative_to(ROOT)),
                    environment='Existing recorded environment; fresh installation not tested',
                    checks=checks)
    with archive.with_suffix('.smoke.json').open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    smoke((ROOT / args.archive).resolve(), args.run)
