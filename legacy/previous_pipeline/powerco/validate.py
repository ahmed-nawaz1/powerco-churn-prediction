"""Read-only consistency validation; does not fit or optimise models."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from powerco.workflow import ROOT, sha, metrics


def validate(run):
    out = Path(run)
    if not out.is_absolute():
        out = ROOT / out
    manifest = json.loads((out / 'manifest.json').read_text())
    assert 'completed_utc' in manifest, 'Run is incomplete'
    for path, expected in json.loads((out / 'checksums.json').read_text()).items():
        assert sha(out / path) == expected, f'Changed result: {path}'
    for path, expected in manifest['input_sha256'].items():
        assert sha(ROOT / path) == expected, f'Changed input: {path}'
    for path, expected in manifest['source_sha256'].items():
        assert sha(ROOT / path) == expected, f'Changed executed source: {path}'
    split = pd.read_csv(out / 'split_assignments.csv')
    assert split.id.is_unique and split.row_index.is_unique
    train = split[split.partition.eq('train')]
    test = split[split.partition.eq('legacy_provisional_test')]
    assert len(train) + len(test) == 14606
    assert not set(train.id) & set(test.id)
    assert set(train.validation_fold) == set(range(5))
    assert test.validation_fold.eq(-1).all()
    fold = pd.read_csv(out / 'fold_metrics.csv')
    assert fold.groupby('model').size().eq(5).all()
    assert len(pd.read_csv(out / 'search_results.csv')) == 12
    lock = json.loads((out / 'selection.json').read_text())
    assert lock['selected_before_new_test'] is True
    pred = pd.read_csv(out / 'provisional_test_predictions.csv')
    assert set(pred.row_index) == set(test.row_index)
    actual = pd.read_csv(out / 'provisional_test_metrics.csv')
    row = actual[(actual.model == lock['model']) & (actual.operating_point == 'training_selected_threshold')].iloc[0]
    recalculated = metrics(pred.churn, pred.probability, lock['threshold'])
    for key, value in recalculated.items():
        assert np.isclose(row[key], value, rtol=1e-12, atol=1e-12), key
    model = joblib.load(out / 'selected_model.joblib')
    assert model['threshold'] == lock['threshold']
    data = pd.read_csv(ROOT / 'client_data.csv').loc[pred.row_index]
    assert np.allclose(model['pipeline'].predict_proba(data)[:, 1], pred.probability, rtol=1e-12, atol=1e-12)
    assert len(list((out / 'figures').glob('*.png'))) == 10
    assert (out / 'git-log.txt').is_file()
    return {'status': 'PASS', 'checksummed_result_files': len(json.loads((out / 'checksums.json').read_text())),
            'split_rows': len(split), 'models': len(fold.model.unique()), 'selected': lock['model'],
            'test_f1_recomputed': recalculated['f1'], 'test_status': 'PROVISIONAL',
            'checks': ['input/source/output checksums', 'unique disjoint split and five folds',
                       '12 search candidates', 'test metrics recomputed from saved probabilities',
                       'reloaded model predictions match', 'ten figures present']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--save', action='store_true', help='Save validation.json (new file only)')
    args = parser.parse_args()
    result = validate(args.run)
    if args.save:
        folder = Path(args.run)
        if not folder.is_absolute():
            folder = ROOT / folder
        with (folder / 'validation.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
