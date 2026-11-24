"""Execute the locked protocol. New output folders only; never import main.py."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, brier_score_loss)
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_validate, cross_val_predict, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from threadpoolctl import threadpool_limits

from powerco.features import SnapshotFeatures, CATEGORICAL, NUMERIC, ENGINEERED, RAW_FEATURES, signed_log

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SCORES = {'precision': 'precision', 'recall': 'recall', 'f1': 'f1',
          'roc_auc': 'roc_auc', 'average_precision': 'average_precision'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), message, flush=True)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def pipeline(model, linear=False):
    numeric = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if linear:
        numeric += [('log', FunctionTransformer(signed_log, feature_names_out='one-to-one')),
                    ('scale', StandardScaler())]
    pre = ColumnTransformer([
        ('num', Pipeline(numeric), NUMERIC + [ENGINEERED]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)
    ])
    return Pipeline([('features', SnapshotFeatures()), ('preprocess', pre), ('model', model)])


def metrics(y, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, prob) if len(np.unique(y)) == 2 else None,
                average_precision=average_precision_score(y, prob) if np.sum(y) else None,
                brier=brier_score_loss(y, prob), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                n=len(y), positives=int(np.sum(y)), threshold=float(threshold))


def structure(c, p, out):
    info = {}
    for name, data, keys in [('clients', c, ['id']), ('prices', p, ['id', 'price_date'])]:
        info[name] = dict(rows=len(data), columns=len(data.columns), unique_ids=data.id.nunique(),
                          row_duplicates=int(data.duplicated().sum()), key_duplicates=int(data.duplicated(keys).sum()),
                          missing=data.isna().sum().to_dict(), dtypes=data.dtypes.astype(str).to_dict(),
                          literal_missing={col: int(data[col].eq('MISSING').sum()) for col in data.select_dtypes(include=['object', 'str'])})
        assert data.duplicated(keys).sum() == 0, f'{name}: duplicate keys need review; nothing silently deleted'
    assert set(c.churn.unique()) == {0, 1}
    dates = {}
    for col in ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']:
        d = pd.to_datetime(c[col], errors='coerce')
        dates[col] = dict(min=str(d.min()), max=str(d.max()), invalid=int(d.isna().sum()))
    d = pd.to_datetime(p.price_date, errors='coerce')
    dates['price_date'] = dict(min=str(d.min()), max=str(d.max()), invalid=int(d.isna().sum()))
    info['date_ranges'] = dates
    info['cutoff'] = 'NOT ESTABLISHED; data ranges are not prediction time'
    info['target_counts'] = c.churn.value_counts().sort_index().to_dict()
    info['price_only_ids'] = len(set(p.id) - set(c.id))
    info['clients_without_price'] = len(set(c.id) - set(p.id))
    write_json(out / 'data_quality.json', info)
    return info


def eda(train, prices, out):
    """Relationships use the legacy training rows only; prices are descriptive."""
    figs = out / 'figures'
    train.describe(include='all').T.to_csv(out / 'training_summary.csv')
    numeric = train.select_dtypes(include=np.number).drop(columns='churn')
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    outliers = pd.DataFrame({'iqr_low': q1 - 1.5 * iqr, 'iqr_high': q3 + 1.5 * iqr,
                            'outside_iqr_count': ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum(),
                            'negative_count': (numeric < 0).sum(), 'zero_count': (numeric == 0).sum()})
    outliers.to_csv(out / 'outlier_diagnostics.csv')
    corr = train.select_dtypes(include=np.number).corr(method='spearman')
    corr.to_csv(out / 'spearman_correlations.csv')
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr)), corr.columns, fontsize=6)
    plt.colorbar(label='Spearman correlation'); plt.title('Training data correlations (association only)')
    savefig(figs / '02_correlations.png')
    train.churn.value_counts().sort_index().plot.bar(color=['#287891', '#df8646'], rot=0)
    plt.xticks([0, 1], ['Retained', 'Churned']); plt.ylabel('Customers'); plt.title('Training class imbalance')
    savefig(figs / '01_churn.png')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, col in zip(axes, ['cons_12m', 'num_years_antig', 'pow_max']):
        for target in [0, 1]:
            v = train.loc[train.churn.eq(target), col]
            if col == 'cons_12m':
                v = np.sign(v) * np.log1p(np.abs(v))
            ax.hist(v, bins=30, density=True, alpha=.5, label=f'churn={target}')
        ax.set_title(col + (' (signed log1p)' if col == 'cons_12m' else ''))
        ax.set_ylabel('Density'); ax.legend(fontsize=8)
    savefig(figs / '03_distributions.png')
    price_cols = [x for x in prices if x.startswith('price_') and x != 'price_date']
    aggregate = prices.groupby('id')[price_cols].agg(['mean', 'std', 'min', 'max'])
    aggregate.columns = ['eda_' + '_'.join(x) for x in aggregate.columns]
    merged = train.merge(aggregate, on='id', how='left', validate='one_to_one')
    assert len(merged) == len(train)
    merged[['id', 'churn'] + list(aggregate.columns)].to_csv(out / 'pricing_exploratory_merge.csv', index=False)
    # Rank-based bins avoid unstable ties; duplicates are collapsed, never force fabricated bins.
    mean_price = 'eda_price_off_peak_var_mean'
    merged['price_bin'] = pd.qcut(merged[mean_price], 4, duplicates='drop')
    relationship = merged.groupby('price_bin', observed=True).churn.agg(['size', 'sum', 'mean'])
    relationship.to_csv(out / 'pricing_relationship.csv')
    relationship['mean'].plot.bar(rot=25, color='#287891', figsize=(8, 4))
    plt.ylabel('Observed churn proportion'); plt.xlabel('Mean off-peak variable price quartile')
    plt.title('Training pricing association - EDA ONLY, cutoff unknown')
    savefig(figs / '04_pricing_eda.png')
    daily = prices[prices.id.isin(train.id)].groupby('price_date')[price_cols].mean()
    daily.to_csv(out / 'price_history.csv')
    return relationship


def segments(raw, y, prob, threshold, out):
    data = raw.copy()
    data['tenure_band'] = pd.cut(data.num_years_antig, [-np.inf, 3, 6, np.inf], labels=['<=3 years', '4-6 years', '>6 years'])
    rows = []
    for col in ['has_gas', 'channel_sales', 'tenure_band']:
        for value, idx in data.groupby(col, observed=True).groups.items():
            pos = data.index.get_indexer(idx)
            row = metrics(np.asarray(y)[pos], np.asarray(prob)[pos], threshold)
            row.update(segment=col, value=str(value), small_positive_sample=row['positives'] < 30)
            row['false_positive_rate'] = row['fp'] / max(1, row['fp'] + row['tn'])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'segment_oof_metrics.csv', index=False)


def explain(selected, X, y, cv, threshold, out, jobs):
    # Fixed first fold: explanations are diagnostic, not an independent performance test.
    tr, va = cv[0]
    model = clone(selected).fit(X.iloc[tr], y.iloc[tr])
    raw = X.iloc[va]
    yp = y.iloc[va]
    importance = permutation_importance(model, raw[RAW_FEATURES], yp, scoring='average_precision',
                                       n_repeats=5, random_state=SEED, n_jobs=jobs)
    frame = pd.DataFrame({'feature': RAW_FEATURES, 'mean_ap_decrease': importance.importances_mean,
                          'std_over_permutations': importance.importances_std}).sort_values('mean_ap_decrease', ascending=False)
    frame.to_csv(out / 'permutation_importance.csv', index=False)
    frame.sort_values('mean_ap_decrease').plot.barh(x='feature', y='mean_ap_decrease',
                xerr='std_over_permutations', legend=False, figsize=(8, 5), color='#287891')
    plt.xlabel('Decrease in validation average precision'); plt.title('Fixed fold 1: raw-feature permutation importance')
    savefig(out / 'figures' / '09_importance.png')
    prob = model.predict_proba(raw)[:, 1]
    pred = prob >= threshold
    errors = np.where(pred, np.where(yp.to_numpy() == 1, 'TP', 'FP'), np.where(yp.to_numpy() == 1, 'FN', 'TN'))
    rows = []
    # One observed example of each available outcome; no invented examples.
    for outcome in ['TP', 'FP', 'FN', 'TN']:
        candidates = np.flatnonzero(errors == outcome)
        if not len(candidates):
            continue
        i = int(candidates[np.argmax(prob[candidates])])
        example = raw.iloc[[i]].copy()
        for col in RAW_FEATURES:
            changed = example.copy()
            replacement = X.iloc[tr][col].mode().iloc[0] if col in CATEGORICAL else X.iloc[tr][col].median()
            changed[col] = replacement
            alternative = float(model.predict_proba(changed)[0, 1])
            rows.append(dict(row_index=int(example.index[0]), outcome=outcome, feature=col,
                             observed_value=str(example[col].iloc[0]), replacement=str(replacement),
                             original_score=float(prob[i]), replacement_score=alternative,
                             score_change=alternative - float(prob[i])))
    pd.DataFrame(rows).to_csv(out / 'local_prediction_sensitivity.csv', index=False)


def run(args):
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)  # no overwrite, including accidental reruns
    (out / 'figures').mkdir()
    (out / 'source').mkdir()
    for path in [ROOT / 'main.py', ROOT / 'requirements.txt', *sorted((ROOT / 'powerco').glob('*.py')),
                 ROOT / 'audit/PROTOCOL.md']:
        shutil.copy2(path, out / 'source' / path.name)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    platform=platform.platform(), seed=SEED, command=sys.argv, jobs=args.jobs,
                    dependencies={k: importlib.metadata.version(k) for k in
                                  ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'joblib', 'threadpoolctl']},
                    input_sha256={f: sha(ROOT / f) for f in ['main.py', 'client_data.csv', 'price_data.csv',
                                                            'cps7008-assessment-1.pdf', 'audit/PROTOCOL.md']},
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'powerco').glob('*.py'))},
                    test_status='LEGACY TEST ALREADY INSPECTED; ALL NEW TEST RESULTS PROVISIONAL',
                    temporal_status='UNVERIFIED SNAPSHOT; PRICING/DATE/FORECAST/FINANCIAL FIELDS EXCLUDED')
    write_json(out / 'manifest.json', manifest)
    frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (out / 'environment-freeze.txt').write_text(frozen.stdout, encoding='utf-8')
    git = subprocess.run(['git', 'log', '--date=iso-strict', '--format=fuller', '--stat'], cwd=ROOT,
                         capture_output=True, text=True)
    (out / 'git-log.txt').write_text(git.stdout if git.returncode == 0 else
        'No Git repository was present at audit time. No earlier commits can be recovered here.\n' + git.stderr,
        encoding='utf-8')
    c = pd.read_csv(ROOT / 'client_data.csv')
    p = pd.read_csv(ROOT / 'price_data.csv')
    quality = structure(c, p, out)
    X, y = c.drop(columns='churn'), c.churn
    tr, te = train_test_split(np.arange(len(c)), test_size=.2, stratify=y, random_state=SEED)
    assert not set(tr) & set(te)
    split = c[['id']].copy()
    split['partition'] = 'train'; split.loc[te, 'partition'] = 'legacy_provisional_test'
    split['row_index'] = split.index
    Xt, yt = X.iloc[tr], y.iloc[tr]
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xt, yt))
    split['validation_fold'] = -1
    for fold, (_, va) in enumerate(cv):
        split.loc[Xt.iloc[va].index, 'validation_fold'] = fold
    split.to_csv(out / 'split_assignments.csv', index=False)
    log('Structure verified; producing training-only exploratory figures')
    eda(c.iloc[tr], p, out)
    selected_fields = []
    for col in c.columns:
        selected_fields.append(dict(feature=col, use=('target' if col == 'churn' else
            'conditional supplied historical snapshot feature' if col in RAW_FEATURES else
            'join/split key only' if col == 'id' else 'excluded: as-of timing unverified'),
            timing_verified=False))
    pd.DataFrame(selected_fields).to_csv(out / 'feature_eligibility.csv', index=False)
    models = {
        'Dummy': pipeline(DummyClassifier(strategy='prior')),
        'LogisticRegression': pipeline(LogisticRegression(class_weight='balanced', max_iter=3000, random_state=SEED), True),
        'RandomForest': pipeline(RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                    class_weight='balanced', random_state=SEED, n_jobs=1)),
        'HistGradientBoosting': pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                                    max_leaf_nodes=15, random_state=SEED)),
        'BalancedHistGradientBoosting': pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                                    max_leaf_nodes=15, class_weight='balanced', random_state=SEED)),
        'SimplerRandomForest': pipeline(RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                    class_weight='balanced', random_state=SEED, n_jobs=1))}
    search = RandomizedSearchCV(pipeline(RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1)),
                {'model__n_estimators': [100, 200, 300], 'model__max_depth': [6, 10, 14, None],
                 'model__min_samples_leaf': [2, 5, 10], 'model__max_features': ['sqrt', .7]},
                n_iter=12, scoring='f1', cv=cv, random_state=SEED, n_jobs=args.jobs, refit=True,
                error_score='raise')
    log('Executing fixed 12-candidate RF search on training folds only')
    search.fit(Xt, yt)
    pd.DataFrame(search.cv_results_).to_csv(out / 'search_results.csv', index=False)
    write_json(out / 'search_best.json', dict(params=search.best_params_, mean_f1=search.best_score_))
    models['TunedRandomForest'] = clone(search.best_estimator_)
    folds = []
    for name, estimator in models.items():
        log('Five-fold evaluation: ' + name)
        scores = cross_validate(estimator, Xt, yt, cv=cv, scoring=SCORES,
                                return_train_score=True, n_jobs=args.jobs, error_score='raise')
        for i in range(5):
            row = dict(model=name, fold=i, fit_time=float(scores['fit_time'][i]))
            row.update({key: float(value[i]) for key, value in scores.items() if key.startswith(('test_', 'train_'))})
            folds.append(row)
        pd.DataFrame(folds).to_csv(out / 'fold_metrics.csv', index=False)
    fold_df = pd.DataFrame(folds)
    means = fold_df.groupby('model', sort=False).mean(numeric_only=True).drop(columns='fold')
    means['validation_f1_std'] = fold_df.groupby('model').test_f1.std()
    means['train_validation_f1_gap'] = means.train_f1 - means.test_f1
    means.to_csv(out / 'cv_summary.csv')
    winner = means.test_f1.idxmax()
    log('Training-only model selected: ' + winner + '; generating OOF threshold evidence')
    oof = cross_val_predict(models[winner], Xt, yt, cv=cv, method='predict_proba', n_jobs=args.jobs)[:, 1]
    thresholds = np.round(np.arange(.05, .951, .01), 2)
    curve = pd.DataFrame([metrics(yt, oof, t) for t in thresholds])
    # Stable tie preference: closest to 0.5, then higher threshold.
    curve['distance_to_default'] = abs(curve.threshold - .5)
    chosen = curve.sort_values(['f1', 'distance_to_default', 'threshold'], ascending=[False, True, False]).iloc[0]
    threshold = float(chosen.threshold)
    curve.to_csv(out / 'threshold_oof.csv', index=False)
    pd.DataFrame({'row_index': Xt.index, 'y': yt.to_numpy(), 'probability': oof,
                  'prediction': (oof >= threshold).astype(int)}).to_csv(out / 'oof_predictions.csv', index=False)
    lock = dict(model=winner, threshold=threshold, selected_before_new_test=True,
                selection='training mean CV F1 at 0.5; training pooled OOF F1 threshold',
                cv_selection_bias='Search/model/threshold selection are not nested; OOF performance is optimistic',
                params=models[winner].get_params(deep=False),
                frozen_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'selection.json', lock)
    segments(Xt, yt.to_numpy(), oof, threshold, out)
    log('Producing fixed-fold diagnostic explanations')
    explain(models[winner], Xt, yt, cv, threshold, out, args.jobs)
    transformed = clone(models[winner][:-1]).fit_transform(Xt, yt)
    before = SnapshotFeatures().fit_transform(Xt)
    write_json(out / 'preprocessing_checks.json', dict(raw_rows=len(Xt), selected_raw_features=len(RAW_FEATURES),
                with_engineering=before.shape[1], encoded_shape=list(transformed.shape),
                engineered_missing_before_imputation=int(before[ENGINEERED].isna().sum()),
                missing_after=int(np.isnan(transformed).sum()), all_finite=bool(np.isfinite(transformed).all()),
                policy='Missing numeric medians, OHE categories and LR scale fit within CV folds. Extreme customers retained.'))
    assert np.isfinite(transformed).all()
    log('SELECTION FROZEN. Starting one provisional test evaluation phase; no further selection permitted')
    test_rows, probabilities = [], {}
    fitted = {}
    for name, estimator in models.items():
        model = clone(estimator).fit(Xt, yt)
        prob = model.predict_proba(X.iloc[te])[:, 1]
        fitted[name] = model
        probabilities[name] = prob
        test_rows.append(dict(model=name, operating_point='default', **metrics(y.iloc[te], prob)))
    prob = probabilities[winner]
    test_rows.append(dict(model=winner, operating_point='training_selected_threshold', **metrics(y.iloc[te], prob, threshold)))
    pd.DataFrame(test_rows).to_csv(out / 'provisional_test_metrics.csv', index=False)
    predictions = c.iloc[te][['id'] + RAW_FEATURES + ['churn']].copy()
    predictions['row_index'] = predictions.index
    predictions['probability'] = prob
    predictions['prediction'] = (prob >= threshold).astype(int)
    predictions['outcome'] = np.where(predictions.prediction.eq(1), np.where(predictions.churn.eq(1), 'TP', 'FP'),
                                      np.where(predictions.churn.eq(1), 'FN', 'TN'))
    predictions.to_csv(out / 'provisional_test_predictions.csv', index=False)
    errors = predictions[predictions.outcome.isin(['FP', 'FN'])]
    pd.concat([errors[errors.outcome.eq(k)].sort_values('probability', ascending=k == 'FN').head(5)
               for k in ['FP', 'FN']]).to_csv(out / 'misclassified_examples.csv', index=False)
    joblib.dump(dict(pipeline=fitted[winner], threshold=threshold, model=winner,
                     eligibility='Conditional snapshot; not approved for deployment'), out / 'selected_model.joblib')
    loaded = joblib.load(out / 'selected_model.joblib')
    assert np.array_equal(loaded['pipeline'].predict_proba(X.iloc[te])[:, 1], prob)
    plt.figure(figsize=(7, 4))
    plot = means[['train_f1', 'test_f1']]
    plot.plot.bar(ax=plt.gca()); plt.ylabel('F1 at 0.5'); plt.xticks(rotation=30, ha='right')
    plt.title('Training/CV gap - selected/tuned CV estimates are optimistic')
    savefig(out / 'figures' / '05_overfitting.png')
    plt.figure(figsize=(7, 4))
    for col in ['precision', 'recall', 'f1']:
        plt.plot(curve.threshold, curve[col], label=col)
    plt.axvline(threshold, color='black', ls='--', label=f'Frozen threshold {threshold:.2f}')
    plt.xlabel('Threshold'); plt.ylabel('Pooled OOF metric'); plt.legend()
    plt.title('Training threshold selection (selection-biased evidence)')
    savefig(out / 'figures' / '06_threshold.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, prob_m in probabilities.items():
        fpr, tpr, _ = roc_curve(y.iloc[te], prob_m)
        precision, recall, _ = precision_recall_curve(y.iloc[te], prob_m)
        axes[0].plot(fpr, tpr, label=name); axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], 'k--'); axes[0].set(xlabel='False positive rate', ylabel='True positive rate')
    axes[1].axhline(y.iloc[te].mean(), color='black', ls='--'); axes[1].set(xlabel='Recall', ylabel='Precision')
    axes[1].legend(fontsize=6); fig.suptitle('Legacy test ROC and PR curves - PROVISIONAL')
    savefig(out / 'figures' / '07_test_curves.png')
    cm = confusion_matrix(y.iloc[te], prob >= threshold, labels=[0, 1])
    plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.xticks([0, 1], ['Retain', 'Churn']); plt.yticks([0, 1], ['Retain', 'Churn'])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'{winner}: provisional test, threshold {threshold:.2f}')
    savefig(out / 'figures' / '08_confusion.png')
    # Calibration diagnostics without applying new test-driven calibration.
    cal = pd.DataFrame({'score': prob, 'y': y.iloc[te].to_numpy()})
    cal['bin'] = pd.cut(cal.score, np.linspace(0, 1, 11), include_lowest=True)
    cal.groupby('bin', observed=True).agg(n=('y', 'size'), observed=('y', 'mean'), mean_score=('score', 'mean')).to_csv(out / 'calibration_diagnostic.csv')
    # Descriptive recall/selection rates; no protected attributes or causal fairness claim.
    seg = pd.read_csv(out / 'segment_oof_metrics.csv')
    view = seg[seg.segment.eq('has_gas')]
    view.plot.bar(x='value', y=['recall', 'false_positive_rate'], figsize=(6, 4))
    plt.ylabel('Rate'); plt.title('OOF service-segment diagnostics (not protected-group fairness)')
    savefig(out / 'figures' / '10_segments.png')
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['selected'] = winner
    manifest['validation'] = dict(disjoint_split=True, unique_keys=True, finite_preprocessing=True,
                                  model_reload_exact=True, selection_before_test=True)
    write_json(out / 'manifest.json', manifest)
    files = {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'checksums.json'}
    write_json(out / 'checksums.json', files)
    log('COMPLETE: ' + str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New run directory; refuses overwrite')
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    # Native thread cap also propagates to joblib child processes.
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    with threadpool_limits(limits=1):
        run(args)
