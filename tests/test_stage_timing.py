"""Notebook新增耗时模块回归测试；小型夹具仅检查软件，不作为HAR成绩。"""
import copy
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_core import fixture, small_config, ROOT
sys.path.insert(0,str(ROOT/'src'))
from har_study.study import make_pipeline, metrics
from har_study.stage_timing import validate_timing_config, summarize_stages, benchmark_stages, verify_stage_timing

class StageTimingTests(unittest.TestCase):
    def test_default_config(self):
        validate_timing_config(json.loads((ROOT/'timing_config.json').read_text()))
    def test_repeated_batches_rejected(self):
        with self.assertRaises(ValueError):
            validate_timing_config(dict(batch_sizes=[1,1], repeats=2, warmups=1, seed=42))
    def test_invalid_repeat_rejected(self):
        with self.assertRaises(ValueError):
            validate_timing_config(dict(batch_sizes=[1,8], repeats=0, warmups=1, seed=42))
    def test_medians(self):
        frame=pd.DataFrame([dict(selection_id='A',batch_size=2,stage='whole',repeat=i,total_ms=v,ms_per_window=v/2)
                            for i,v in enumerate([1.,2.,3.])])
        r=summarize_stages(frame).iloc[0]
        self.assertEqual(r.median_ms,2.);self.assertEqual(r.ms_per_window,1.);self.assertEqual(r.n_repeats,3)
    def test_batch_larger_than_data_rejected(self):
        d=fixture([1,2]); p=make_pipeline('LogisticRegression','full',24,small_config()).fit(d.X,d.y)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                benchmark_stages(Path(td),d.X,{'LogisticRegression__full':p},
                                 dict(batch_sizes=[1000], repeats=2,warmups=1,seed=42))
    def test_integration_and_recalculation(self):
        import joblib
        d=fixture([1,2,3]); c=small_config()
        models={}; scores=[]; selections=[]
        for method,k in [('full',24),('pca',8),('anova',8)]:
            sid='LogisticRegression__'+method
            p=make_pipeline('LogisticRegression',method,k,c).fit(d.X,d.y)
            models[sid]=p
            scores.append(dict(selection_id=sid,model='LogisticRegression',method=method,k=k,
                               **metrics(d.y,p.predict(d.X))))
            selections.append(dict(selection_id=sid,meets_cv_tolerance=True))
        config=dict(batch_sizes=[1,8,len(d.y)],repeats=3,warmups=1,seed=42)
        with tempfile.TemporaryDirectory() as td,redirect_stdout(io.StringIO()):
            out=Path(td);(out/'tables').mkdir();(out/'models').mkdir()
            for sid,p in models.items():joblib.dump(p,out/'models'/f'{sid}.joblib')
            pd.DataFrame(scores).to_csv(out/'tables/test_metrics.csv',index=False)
            pd.DataFrame(selections).to_csv(out/'tables/selected_configs.csv',index=False)
            (out/'timing_config_frozen.json').write_text(json.dumps(config))
            summary=benchmark_stages(out,d.X,models,config)
            result=verify_stage_timing(out)
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['n_raw_rows'],108)
            raw=pd.read_csv(out/'tables/stage_timing_raw.csv')
            zeros=raw[raw.selection_id.str.endswith('__full') & (raw.stage=='reduce')]
            self.assertTrue(zeros.total_ms.eq(0).all());self.assertFalse(zeros.measured.any())
            ratios=pd.read_csv(out/'tables/PCA_vs_full_with_timing_breakdown.csv')
            self.assertEqual(len(ratios),3);self.assertTrue(ratios.pca_speedup_full_div_pca.gt(0).all())
            summary.loc[0,'median_ms']+=1
            summary.to_csv(out/'tables/stage_timing_summary.csv',index=False)
            with self.assertRaises(AssertionError): verify_stage_timing(out)

if __name__=='__main__': unittest.main()
