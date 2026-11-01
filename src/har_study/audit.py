"""从保存结果派生数值对比表；不重新选参数，不生成报告。"""
from pathlib import Path
import json
import pandas as pd

def decision_summary(run: Path) -> pd.DataFrame:
    """派生摘要，明确分开训练期达标与测试期性能，不能把诊断失败当成功压缩。"""
    run = Path(run)
    tables = run / 'tables'
    selected = pd.read_csv(tables / 'selected_configs.csv')
    test = pd.read_csv(tables / 'test_metrics.csv')
    sizes = pd.read_csv(tables / 'final_fit_and_size.csv').set_index('selection_id')
    timing = pd.read_csv(tables / 'timing_summary.csv').set_index(['selection_id', 'kind'])
    config = json.loads((run / 'config_frozen.json').read_text(encoding='utf-8'))
    rows = []
    for record in selected.to_dict('records'):
        sid, model = record['selection_id'], record['model']
        result = test.loc[test.selection_id == sid].iloc[0]
        base = test.loc[(test.model == model) & (test.method == 'full')].iloc[0]
        base_sid = base.selection_id
        gap = 100 * (result.macro_f1 - base.macro_f1)
        rows.append({
            'selection_id': sid, 'model': model, 'method': record['method'], 'k': int(record['k']),
            'cv_macro_f1_percent': 100 * record['cv_macro_f1'],
            'cv_tolerance_met': bool(record['meets_cv_tolerance']),
            'selection_rule': record['selection_rule'],
            'test_accuracy_percent': 100 * result.accuracy,
            'test_macro_f1_percent': 100 * result.macro_f1,
            'test_delta_from_full_pp': gap,
            'test_within_tolerance_descriptive': bool(gap >= -100 * config['allowed_macro_f1_drop'] - 1e-9),
            'dimension_reduction_percent': 100 * (1 - result.k / 561),
            'batch_ms_per_window': timing.loc[(sid, 'batch'), 'median_ms_per_sample'],
            'single_ms_per_call': timing.loc[(sid, 'single'), 'median_ms_per_sample'],
            'batch_speedup_over_same_classifier_full': (
                timing.loc[(base_sid, 'batch'), 'median_ms_per_sample'] / timing.loc[(sid, 'batch'), 'median_ms_per_sample']),
            'single_speedup_over_same_classifier_full': (
                timing.loc[(base_sid, 'single'), 'median_ms_per_sample'] / timing.loc[(sid, 'single'), 'median_ms_per_sample']),
            'pipeline_mib': sizes.loc[sid, 'pipeline_bytes'] / 1024**2,
            'pipeline_size_change_percent': 100 * (sizes.loc[sid, 'pipeline_bytes'] / sizes.loc[base_sid, 'pipeline_bytes'] - 1),
        })
    return pd.DataFrame(rows)


