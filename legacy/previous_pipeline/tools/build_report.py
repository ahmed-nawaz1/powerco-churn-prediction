"""Build an evidence-linked working draft from an already validated run.

Does not train models, choose parameters or revisit test selection.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, KeepTogether, PageBreak)

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def build(run, out):
    assert (run / 'validation.json').is_file(), 'Run validation and save validation.json first'
    assert json.loads((run / 'validation.json').read_text())['status'] == 'PASS'
    # Result checksums are inspected again without changing evidence.
    for path, digest in json.loads((run / 'checksums.json').read_text()).items():
        assert hashlib.sha256((run / path).read_bytes()).hexdigest() == digest, path
    out.mkdir(parents=True, exist_ok=False)
    quality = json.loads((run / 'data_quality.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    lock = json.loads((run / 'selection.json').read_text())
    before = json.loads((run / 'preprocessing_checks.json').read_text())
    cv = read_csv(run / 'cv_summary.csv')
    test = read_csv(run / 'provisional_test_metrics.csv')
    pricing = read_csv(run / 'pricing_relationship.csv')
    importance = read_csv(run / 'permutation_importance.csv')
    local = read_csv(run / 'local_prediction_sensitivity.csv')
    seg = read_csv(run / 'segment_oof_metrics.csv')
    errors = read_csv(run / 'misclassified_examples.csv')
    outliers = read_csv(run / 'outlier_diagnostics.csv')
    final = next(x for x in test if x['operating_point'] == 'training_selected_threshold')
    baseline = next(x for x in test if x['model'] == lock['model'] and x['operating_point'] == 'default')
    selected_cv = next(x for x in cv if x['model'] == lock['model'])
    simple = next(x for x in cv if x['model'] == 'SimplerRandomForest')
    tuned = next(x for x in cv if x['model'] == 'TunedRandomForest')
    hgb = next(x for x in cv if x['model'] == 'HistGradientBoosting')
    balanced_hgb = next(x for x in cv if x['model'] == 'BalancedHistGradientBoosting')
    gas = [x for x in seg if x['segment'] == 'has_gas']
    low, high = pricing[0], pricing[-1]
    annual_outliers = next(x for x in outliers if x[''] == 'cons_12m')
    local_example = max(local, key=lambda x: abs(number(x, 'score_change')))
    positive = [x for x in importance if number(x, 'mean_ap_decrease') > 0]
    top_names = ', '.join(x['feature'] for x in positive[:3]) or 'no consistently positive feature'
    count_churn = quality['target_counts']['1']
    prevalence = count_churn / quality['clients']['rows']
    name = lock['model']
    threshold = lock['threshold']
    train_n, test_n = before['raw_rows'], int(number(final, 'n'))
    sections = []
    def add(title, *paragraphs):
        sections.append((title, list(paragraphs)))

    add('1. Introduction and Problem Framing',
        f'PowerCo wishes to identify SME customers who may leave within the following three months. '
        f'The supplied churn label makes this a supervised binary classification task: 1 denotes departure and 0 retention. '
        f'The operational concept is a customer snapshot scored before that outcome window. Unfortunately, no verified '
        f'snapshot or prediction date is supplied. Therefore the experiment is conditional research rather than evidence '
        f'of a deployment-ready forecasting service. Its objectives are to compare discrimination, explain model behaviour '
        f'and identify questions for a retention pilot, while recognising that prediction does not establish preventable churn.',
        f'The files contain {quality["clients"]["rows"]:,} customer records and {quality["prices"]["rows"]:,} price observations. '
        f'The working assumption is that selected historical consumption and contract descriptors were available at the '
        f'customer snapshot. IDs are linkage keys, never predictors. Raw dates, historical pricing, forecasts and financial '
        f'fields are withheld from primary models until as-of availability is verified. This policy sacrifices possible signal '
        f'to make the uncertainty explicit; it does not prove every excluded variable leaks.',
        'A false positive can consume contact capacity, trigger an unnecessary discount or annoy a loyal customer. '
        'A false negative can miss a possible retention opportunity. Their relative financial cost is unknown, as are '
        'contact budget and treatment effectiveness. F1 is consequently a provisional research objective rather than '
        'an economic optimum. Neither a default threshold nor a churn score authorises automated customer treatment.')

    add('2. Exploratory Data Analysis',
        f'The customer table has {quality["clients"]["columns"]} columns and the price table '
        f'{quality["prices"]["columns"]}. Structural checks found no duplicate customer IDs, duplicate price ID/date keys, '
        f'full-row duplicates or actual null values. However, the literal MISSING appears 3,725 times in channel_sales '
        f'and 64 times in origin_up. A null-only check would miss this distinction. All customers have pricing records; '
        f'{quality["price_only_ids"]:,} price-only IDs have no labelled customer record and are not additional training cases. '
        f'The data dictionary also mentions absent activity_new and inconsistently describes meter-rent duration.',
        f'There are {count_churn:,} churned customers ({prevalence:.2%}) overall. Figure 1 shows the training imbalance. '
        f'A classifier predicting retention for everyone could achieve {1-prevalence:.2%} full-dataset accuracy while detecting '
        f'no churn, which motivates minority-class metrics. After reconstructing the legacy split, all relationship charts '
        f'and correlation calculations use the {train_n:,} training customers. Earlier full-data inspection is still acknowledged; '
        f'this restriction cannot make historical test exposure disappear.',
        f'Annual consumption has {int(number(annual_outliers, "outside_iqr_count")):,} training observations outside '
        f'the conventional 1.5-IQR limits. Such flags identify unusual values, not erroneous customers. Figure 2 uses '
        f'a signed-log consumption scale to reveal the bulk distribution without deleting large consumers. The archived '
        f'Spearman matrix supports inspection of monotonic associations and correlated predictors; it does not establish '
        f'independent or causal effects. Differences between distributions require substantive interpretation beyond visual separation.',
        'The descriptive pricing pattern is not monotonically increasing across the four displayed bins. '
        'Consequently a simple claim that more expensive electricity necessarily predicts more churn is not '
        'supported by this comparison. Binning can conceal within-group variation and correlated customer mix. '
        'The historical price series and all six price families are retained in the exploratory artefact so '
        'this investigation can be extended after timing clarification rather than reduced to a single plot.',
        f'Price dates span January through December 2015. Training customers are joined one-to-one to customer-level '
        f'mean, spread and range price summaries. The lowest and highest mean off-peak variable-price bins have observed '
        f'churn proportions {number(low,"mean"):.2%} and {number(high,"mean"):.2%}, respectively (Figure 3). '
        f'This is a descriptive comparison affected by customer mix, ties and possible timing overlap. Neither a causal '
        f'price effect nor incremental predictive benefit is established. Pricing remains a candidate for modelling once '
        f'the cutoff and observation windows are confirmed.')

    add('3. Data Pre-processing and Feature Engineering',
        'The primary feature policy retains channel_sales, origin_up, has_gas, cons_12m, cons_gas_12m, '
        'cons_last_month, nb_prod_act, num_years_antig and pow_max. These represent acquisition route, service mix, '
        'past consumption and supplied contract state. Their historical interpretation remains conditional on data '
        'provenance. In particular, supplied tenure is not recomputed against an invented calendar date. '
        'The feature eligibility table records every supplied field and its intended role; forecasts and financial '
        'summaries are excluded rather than silently assumed to predate churn.',
        f'A recent-electricity share divides last-month usage by positive annual electricity consumption. Zero or invalid '
        f'denominators produce missing values instead of infinite ratios. This adds a scale-relative signal without using '
        f'target labels or dates. There are {before["engineered_missing_before_imputation"]:,} undefined training shares. '
        f'The transformation begins with nine raw features and produces ten fields before encoding; the fitted selected '
        f'pipeline produces {before["encoded_shape"][1]} numerical columns and no nonfinite output. This before/after '
        f'evidence is saved rather than inferred from successful fitting.',
        'Categorical values use explicit missing states and unknown-safe one-hot encoding. Numerical missing values '
        'use medians learned only from the relevant training fold. Logistic Regression additionally uses a signed '
        'logarithm and standardisation to reduce scale differences and the influence of large consumption values. '
        'Tree models retain numerical scale. Extreme observations are retained because the supplied information '
        'does not distinguish large valid SMEs from data-entry errors. No arbitrary clipping rule or outlier deletion '
        'is justified from test results.',
        'Every learned transformation is inside the estimator pipeline, so encoding, imputation and scaling are '
        'refitted during each CV training fold rather than on validation customers (scikit-learn developers, n.d.-a). '
        'Class weighting addresses imbalance in LR and RF; an explicitly balanced boosting alternative tests weighting '
        'against the original unweighted boosting design. Synthetic oversampling is unnecessary for this comparison '
        'and is not applied across validation boundaries. Focused checks confirm exclusion invariance, finite '
        'zero-denominator handling and unknown-category behaviour.')

    add('4. Model Development and Experimentation',
        f'The original seed-42 stratified 80/20 split is reconstructed, containing {train_n:,} training and {test_n:,} '
        f'legacy-test customers. The training partition uses five shuffled stratified folds with fixed seed 42. '
        f'Each customer appears in one validation fold, and saved assignments allow exact reconstruction. '
        f'Logistic Regression is the interpretable baseline; Random Forest captures interactions; histogram gradient '
        f'boosting provides a sequential nonlinear alternative. A prior-probability dummy establishes a low-skill control. '
        f'Scikit-learn supplies the common estimator interface (Pedregosa et al., 2011).',
        'The declared comparison also includes balanced boosting and a simpler forest. The latter uses 200 trees, '
        'depth six and at least twenty samples per leaf, explicitly trading flexibility for lower variance. '
        'The RF random search evaluates twelve seed-fixed configurations across five folds, varying tree count '
        'between 100 and 300, maximum depth, minimum leaf size and feature subsampling. All candidates, parameters, '
        'fold scores and training times are archived. This search is meaningful optimisation, but it is not an '
        'exhaustive search of every model family.',
        f'Table 1 reports executed means at the default decision rule. {name} is selected by the highest mean '
        f'validation F1, {number(selected_cv,"test_f1"):.4f}, before the new test phase. Tuned-RF train and validation '
        f'F1 are {number(tuned,"train_f1"):.4f} and {number(tuned,"test_f1"):.4f}; simpler-RF values are '
        f'{number(simple,"train_f1"):.4f} and {number(simple,"test_f1"):.4f}. Figure 4 exposes the train/validation '
        f'gap rather than treating training accuracy as success. A smaller gap alone does not justify a model '
        f'with inferior validation behaviour.',
        f'The unweighted boosting model has CV recall {number(hgb,"test_recall"):.4f} and precision '
        f'{number(hgb,"test_precision"):.4f}; balanced boosting changes these to '
        f'{number(balanced_hgb,"test_recall"):.4f} and {number(balanced_hgb,"test_precision"):.4f}. '
        f'This demonstrates the minority-detection versus false-alarm trade-off at the same default threshold. '
        f'Unweighted boosting nevertheless has CV average precision {number(hgb,"test_average_precision"):.4f}, '
        f'compared with {number(tuned,"test_average_precision"):.4f} for tuned RF. Selection by default F1 '
        f'therefore does not imply superiority for every ranking objective. Thresholds are not optimised for '
        f'every candidate, and this limitation leaves room for a separately predeclared future experiment.',
        f'Pooled out-of-fold scores from the selected training candidate choose threshold {threshold:.2f} by maximum '
        f'F1 on a declared 0.05-0.95 grid. Threshold adjustment changes decisions without changing score ranking '
        f'(scikit-learn developers, n.d.-b). The model, parameters and threshold are frozen in selection.json before '
        f'new test predictions. Search, candidate selection and threshold selection reuse training folds, so their '
        f'CV/OOF summaries are optimistically selected, not unbiased nested-CV estimates. No fresh-independent '
        f'validation claim is made, and no test-driven second search is performed.')

    add('5. Model Evaluation and Error Analysis',
        f'Table 2 contains the locked provisional test comparisons, while Figure 5 shows the selected confusion '
        f'matrix. At threshold {threshold:.2f}, the selected model achieves precision {number(final,"precision"):.4f}, '
        f'recall {number(final,"recall"):.4f}, F1 {number(final,"f1"):.4f}, ROC-AUC {number(final,"roc_auc"):.4f} '
        f'and average precision {number(final,"average_precision"):.4f}. Its default-rule F1 is '
        f'{number(baseline,"f1"):.4f}. Average precision summarises the precision-recall curve; it is reported with '
        f'that name rather than silently equated with trapezoidal PR area. ROC and PR plots are also archived.',
        f'The final confusion counts are TN={int(number(final,"tn"))}, FP={int(number(final,"fp"))}, '
        f'FN={int(number(final,"fn"))} and TP={int(number(final,"tp"))}. Thus '
        f'{int(number(final,"fp"))+int(number(final,"tp")):,} of {test_n:,} customers are flagged in this retrospective '
        f'sample, including {int(number(final,"fp"))} customers who actually stayed. This illustrates contact workload '
        f'and missed churn without assigning fictional financial costs. The saved error examples include up to '
        f'five high-score false positives and five low-score false negatives, linked to actual source rows; '
        f'no invented customer story is attached to them.',
        'OOF segment diagnostics examine gas service, sales channel and supplied-tenure bands. They include sample '
        'size, positive count, recall, precision, F1 and false-positive rate, with a warning for groups containing '
        'fewer than thirty positives. These summaries help detect inconsistent service performance and sparse '
        'groups, but they reuse selected-model OOF evidence. They are not independent fairness validation. '
        'Differences may reflect prevalence, data quality or correlated contract characteristics; causal explanations '
        'would require additional evidence.',
        f'For the observed gas-service groups, OOF recall is {number(gas[0],"recall"):.4f} '
        f'(group {gas[0]["value"]}, {gas[0]["positives"]} positives) versus '
        f'{number(gas[1],"recall"):.4f} (group {gas[1]["value"]}, {gas[1]["positives"]} positives). '
        f'The corresponding false-positive rates are {number(gas[0],"false_positive_rate"):.4f} and '
        f'{number(gas[1],"false_positive_rate"):.4f}. These differences warrant checking who receives contact '
        f'and which customers are missed. They do not determine a fair allocation rule without knowing '
        f'customer need, intervention benefit and governance priorities.',
        'The original project had already inspected this same test set. Consequently every new test statistic remains '
        'provisional even though the new selection was frozen first. Replacing it with another random split of '
        'previously exposed customers would not restore independence. Calibration-bin and Brier diagnostics are '
        'saved without fitting a test-derived calibration correction. Class weighting can change score calibration, '
        'so operational probability statements require validation on genuinely new, correctly time-stamped '
        'customers. Neither fold variation nor the displayed single test result proves future reliability.')

    add('6. Interpretability, Fairness and Ethical Considerations',
        f'Fixed-fold validation permutation importance measures the fall in average precision when one raw feature '
        f'is shuffled five times. The largest positive mean decreases involve {top_names}; the full means and '
        f'permutation variability appear in Figure 6 and permutation_importance.csv. This describes reliance by '
        f'the selected model, not an intrinsic importance ranking. Raw inputs are permuted before feature '
        f'engineering, so the recent/annual ratio is recomputed coherently rather than permuted separately. '
        f'Correlated variables can share signal and make individual permutation effects misleading '
        f'(scikit-learn developers, n.d.-c).',
        f'Local sensitivity supplies a complementary prediction-level diagnostic. One actual first-fold example '
        f'from each available TP/FP/FN/TN outcome is compared with a training median or modal replacement for '
        f'each raw field. For source row {local_example["row_index"]}, replacing {local_example["feature"]} '
        f'changes the score from {number(local_example,"original_score"):.4f} to '
        f'{number(local_example,"replacement_score"):.4f}. These are model perturbations, not guaranteed feasible '
        f'customer actions or additive SHAP contributions. Replacing correlated attributes can produce unrealistic '
        f'combinations. The explanation model is trained only on the fixed fold-training rows, although model '
        f'selection previously used the wider training folds.',
        'Hashed channel, campaign and ID strings protect some direct meaning but do not guarantee anonymity. '
        'IDs and row-level predictions should have restricted access, a defined retention period and an authorised '
        'assessment purpose. Row-level examples belong in the controlled software evidence, not a public portfolio. '
        'Customer explanations should describe uncertainty, permit human review and avoid presenting association '
        'as responsibility for churn.',
        'No protected-group attributes are supplied. Service/channel/tenure differences therefore cannot certify '
        'demographic fairness or rule out proxy discrimination. Missing acquisition records may reflect collection '
        'processes rather than behaviour. Governance should examine data provenance, treatment allocation, '
        'complaints and unequal outcomes before any pilot. A churn score should not automatically deny essential '
        'services or impose punitive prices. SHAP is an optional technique in the brief; permutation and observed '
        'prediction sensitivity provide explicit, limited explanations without claiming causal understanding.')

    add('7. Business Recommendations and Critical Evaluation',
        f'{name} is the provisional research candidate because it wins the declared training-CV F1 comparison, '
        f'with its observed complexity, gap and error profile available for review. Logistic Regression remains '
        f'a transparent comparator; a performance advantage does not automatically outweigh operational '
        f'interpretability or maintenance requirements. The {threshold:.2f} threshold is a research setting. '
        f'Business adoption depends on verified budgets, error costs, score calibration and intervention effects, '
        f'none of which the supplied files establish.',
        'The first action should be a provenance review: establish customer snapshot time, the three-month label '
        'window, forecast creation dates and when prices and margin records became available. Then reconstruct '
        'only pre-cutoff pricing summaries and compare customer-only with customer-plus-price models under '
        'the same temporal validation design. The present pricing analysis gives a question to test; it neither '
        'justifies permanent price exclusion nor supports a claim that higher prices cause departures.',
        'After validation on genuinely new customers, a governed pilot could rank customers for human-reviewed '
        'retention contact. Agree contact capacity, eligible offers and customer safeguards first. Randomised '
        'offer/control groups are needed to estimate incremental retention; contacting high-risk customers '
        'does not demonstrate that contact prevented churn. Calculate savings only after observing response, '
        'offer expense and margin implications. No monetary benefit or return on investment is estimated here.',
        'Monitoring should track feature and score distributions, missing/unknown categories, service-segment '
        'contact rates and complaints. Once three-month outcomes mature, reassess recall, precision, F1, '
        'ROC-AUC, average precision and calibration. Alert limits should be agreed from an approved baseline, '
        'not invented as universal constants. Retraining requires a versioned data snapshot, frozen selection '
        'rules and independent evaluation. The archived seeds, splits, dependencies, source hashes and saved '
        'pipeline support this review but do not remedy unresolved chronology or guarantee new-environment '
        'installation. The genuine absence of historical Git commits and the student authorship/declaration '
        'review remain visible limitations of the submission evidence.',
        'Earlier user-reported results used a broader customer feature design and lack saved console evidence '
        'in this folder. Their higher scores are preserved as reported history, not reproduced findings. '
        'The current conservative feature experiment answers a different conditional question, so a lower '
        'score is not evidence of an implementation failure or a direct measure of the cost of excluding '
        'prices. Feature eligibility, test exposure and objective choice must be resolved before comparing '
        'approaches as operational alternatives.')

    refs = [
        'Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., '
        'Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., '
        'Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine '
        'Learning Research, 12(85), 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html',
        'scikit-learn developers. (n.d.-a). Common pitfalls and recommended practices (version 1.9.1). '
        'Retrieved November 17, 2026, from https://scikit-learn.org/stable/common_pitfalls.html',
        'scikit-learn developers. (n.d.-b). Tuning the decision threshold for class prediction (version 1.9.1). '
        'Retrieved November 17, 2026, from https://scikit-learn.org/stable/modules/classification_threshold.html',
        'scikit-learn developers. (n.d.-c). Permutation feature importance (version 1.9.1). '
        'Retrieved November 17, 2026, from https://scikit-learn.org/stable/modules/permutation_importance.html',
        'CPS7008. (2026). Machine learning assessment 1 brief and PowerCo data description [Supplied module materials].'
    ]
    body_words = sum(len(re.findall(r"\b[\w'-]+\b", p)) for _, ps in sections for p in ps)
    assert body_words <= 3000, f'Body exceeds maximum: {body_words}'
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=9.3, leading=13.1,
                              spaceAfter=8, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='SmallReport', fontName='Helvetica', fontSize=7.4, leading=10, spaceAfter=5,
                              wordWrap='CJK'))
    styles['Heading1'].fontSize = 14
    styles['Heading1'].leading = 18
    styles['Heading1'].spaceBefore = 12
    styles['Heading1'].keepWithNext = True
    doc = SimpleDocTemplate(str(out / 'powerco_report_WORKING_DRAFT.pdf'), pagesize=(595.28, 841.89),
                            rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=43,
                            title='PowerCo Customer Churn - Working Draft', author='Student review required')
    story = []
    md = ['# PowerCo Customer Churn Prediction', '',
          'AI-ASSISTED WORKING DRAFT - student review, reflection and declaration required.', '',
          f'Run: `{run.relative_to(ROOT)}`. Main prose: {body_words} words; tables, captions, references and appendix excluded from this count.', '',
          'All test results are provisional. No verified prediction cutoff or business costs.', '']
    story += [Paragraph('PowerCo Customer Churn Prediction', styles['Title']),
              Paragraph('CPS7008 Assessment 1 | AI-assisted working draft', styles['Heading2']),
              Paragraph(f'Run: {escape(str(run.relative_to(ROOT)))} | Body prose: {body_words} words.<br/>'
                        'Student review and declarations required; word-count scope unconfirmed.', styles['BodyReport']),
              Paragraph('Cutoff unresolved; test previously inspected; selection non-nested. '
                        'All test results provisional.', styles['BodyReport'])]

    def figure(file, caption, width=490):
        path = run / 'figures' / file
        iw, ih = ImageReader(str(path)).getSize()
        height = width * ih / iw
        if height > 285:
            width *= 285 / height
            height = 285
        story.append(KeepTogether([Image(str(path), width=width, height=height),
                                  Paragraph(escape(caption), styles['SmallReport']), Spacer(1, 6)]))
        md.extend([f'![{caption}](../../{run.relative_to(ROOT).as_posix()}/figures/{file})', '', caption, ''])

    def table(headers, rows, caption, widths):
        cells = [[Paragraph(escape(str(x)), styles['SmallReport']) for x in headers]]
        cells += [[Paragraph(escape(str(x)), styles['SmallReport']) for x in row] for row in rows]
        tab = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcecf0')),
                                 ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f8')]),
                                 ('VALIGN',(0,0),(-1,-1),'TOP'), ('BOTTOMPADDING',(0,0),(-1,-1),5),
                                 ('LINEBELOW',(0,0),(-1,0),.5,colors.grey)]))
        story.append(KeepTogether([Paragraph(escape(caption), styles['SmallReport']), tab, Spacer(1,10)]))
        md.extend([caption, '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'])
        md.extend('| ' + ' | '.join(map(str,row)) + ' |' for row in rows)
        md.append('')

    labels = {'LogisticRegression':'Logistic Regression', 'RandomForest':'Random Forest',
              'HistGradientBoosting':'HGB (unweighted)', 'BalancedHistGradientBoosting':'HGB (balanced)',
              'SimplerRandomForest':'Simpler RF', 'TunedRandomForest':'Tuned RF', 'Dummy':'Dummy'}
    for i, (title, paragraphs) in enumerate(sections, start=1):
        story.append(Paragraph(escape(title), styles['Heading1']))
        md += ['## ' + title, '']
        for p in paragraphs:
            story.append(Paragraph(escape(p), styles['BodyReport'])); md.extend([p, ''])
        if i == 2:
            figure('01_churn.png', 'Figure 1. Training class counts; source: training partition, seed 42.', width=330)
            figure('03_distributions.png', 'Figure 2. Training customer distributions by churn; consumption shown on signed-log scale.')
            figure('04_pricing_eda.png', 'Figure 3. Descriptive price/churn bins; pricing is EDA-only, cutoff unknown.', width=410)
        if i == 4:
            rows = [[labels[x['model']], f'{number(x,"train_f1"):.4f}', f'{number(x,"test_f1"):.4f}',
                     f'{number(x,"test_recall"):.4f}', f'{number(x,"test_roc_auc"):.4f}',
                     f'{number(x,"test_average_precision"):.4f}'] for x in cv]
            table(['Model','Train F1','CV F1','CV recall','CV ROC-AUC','CV AP'], rows,
                  'Table 1. Executed five-fold means at default threshold; selection-biased estimates. Source: cv_summary.csv.',
                  [135,70,70,70,80,70])
            figure('05_overfitting.png', 'Figure 4. Training and CV F1; tuned and selected CV estimates are optimistic.')
        if i == 5:
            rows = [[labels[x['model']] + (' (selected threshold)' if x['operating_point'] != 'default' else ''),
                     f'{number(x,"precision"):.4f}',f'{number(x,"recall"):.4f}',f'{number(x,"f1"):.4f}',
                     f'{number(x,"roc_auc"):.4f}',f'{number(x,"average_precision"):.4f}'] for x in test]
            table(['Model / rule','Precision','Recall','F1','ROC-AUC','AP'], rows,
                  'Table 2. Provisional legacy-test results. Source: provisional_test_metrics.csv.', [155,68,68,68,68,68])
            figure('08_confusion.png', 'Figure 5. Provisional test confusion matrix at frozen training-selected threshold.', width=350)
            rows = [[x['value'], x['n'], x['positives'], f'{number(x,"recall"):.4f}',
                     f'{number(x,"false_positive_rate"):.4f}'] for x in gas]
            table(['Gas service','OOF n','Churn n','Recall','FP rate'], rows,
                  'Table 3. Training OOF service-segment diagnostics; selected/optimistic, not protected-group fairness.',
                  [135,90,90,90,90])
        if i == 6:
            figure('09_importance.png', 'Figure 6. Fixed-fold permutation importance: mean AP decrease and permutation standard deviation.', width=440)
    story.append(Paragraph('8. References', styles['Heading1'])); md += ['## 8. References', '']
    for ref in refs:
        story.append(Paragraph(escape(ref), styles['SmallReport'])); md.extend([ref, ''])
    story.append(Paragraph('9. Appendices and Student Review', styles['Heading1']))
    appendix = [
        'The software contains a complete evidence map in audit/COMPLIANCE_AUDIT.md. '
        'manifest.json, split_assignments.csv and checksums.json identify source, data and outputs. '
        'legacy_reported_results.json preserves unverified earlier scores.',
        'Student tasks: confirm AI permission; read sources; review authorship; complete reflection/declaration; '
        'seek cutoff clarification and genuine prior Git history. Submit the reviewed report separately from '
        'the software ZIP. Do not submit this working draft unchanged.'
    ]
    md += ['## 9. Appendices and Student Review', '']
    for p in appendix:
        story.append(Paragraph(escape(p), styles['SmallReport'])); md.extend([p, ''])
    def footer(canvas, document):
        canvas.saveState(); canvas.setFont('Helvetica',7)
        canvas.setFillColor(colors.HexColor('#52646d'))
        canvas.drawString(45,24,'CPS7008 | WORKING DRAFT')
        canvas.drawRightString(550,24,f'Page {document.page}'); canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (out / 'report_draft.md').write_text('\n'.join(md), encoding='utf-8')
    (out / 'report_metadata.json').write_text(json.dumps(dict(body_word_count=body_words,
           run=str(run.relative_to(ROOT)), status='WORKING DRAFT - STUDENT REVIEW REQUIRED',
           figure_count=6, tables=3, references_verified='17 November 2026'), indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(out), body_words=body_words), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build((ROOT / args.run).resolve(), (ROOT / args.output).resolve())
