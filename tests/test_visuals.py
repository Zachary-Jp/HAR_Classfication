"""简洁图表、色带和标签回归测试；夹具不作为实验结果。"""
import sys
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from har_study.plots import (check_layout, configure_font, canvas, confusion_figure,
                            heatmap_cells, parameter_figure, _sorted, MODEL_ORDER)
import matplotlib.pyplot as plt

class VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: configure_font()
        except RuntimeError: pass
    def tearDown(self): plt.close('all')
    def test_confusion_white_to_dark_blue(self):
        counts = np.eye(6, dtype=int) * 30
        fig = confusion_figure(counts, 'Test confusion')
        image = fig.axes[0].images[0]
        self.assertEqual(image.get_cmap().name, 'Blues')
        self.assertIsNone(image.get_alpha())
        self.assertGreater(np.mean(image.cmap(0.)[:3]), .95)
        self.assertLess(np.mean(image.cmap(1.)[:3]), .35)
        np.testing.assert_array_equal(image.get_array(), counts)
    def test_no_lower_left_note(self):
        fig, ax = canvas('Title')
        self.assertEqual(len(fig.texts), 1)
        self.assertGreater(fig.texts[0].get_position()[1], .9)
    def test_counts_only_in_matrix(self):
        fig = confusion_figure(np.eye(6, dtype=int), 'Test')
        self.assertEqual(len(fig.axes[0].texts), 36)
        self.assertTrue(all(t.get_text().isdigit() for t in fig.axes[0].texts))
    def test_shared_count_color_range(self):
        fig = confusion_figure(np.eye(6), 'Test', vmax=500)
        self.assertEqual(fig.axes[0].images[0].get_clim(), (0., 500.))
    def test_fisher_parameter_heatmap(self):
        config = json.loads((ROOT/'config.json').read_text())
        df=pd.DataFrame([{'params_json':'{"model__shrinkage":null}', 'cv_macro_f1':.91},
                         {'params_json':'{"model__shrinkage":"auto"}', 'cv_macro_f1':.92}])
        values, ys, xs, yl, xl = heatmap_cells(df, 'FisherLDA', config)
        np.testing.assert_allclose(values[0, :2], [91.,92.])
        self.assertTrue(np.isnan(values[0, 2:]).all())  # 未提供的数值参数保持空白，不伪造得分。
        self.assertEqual(xs, ['不收缩', '自动收缩', '0.05', '0.2', '0.5'])
    def test_missing_heatmap_cell_not_zero(self):
        config=json.loads((ROOT/'config.json').read_text())
        df=pd.DataFrame([{'params_json':'{"model__C":1.0}', 'cv_macro_f1':.9}])
        values,*_ = heatmap_cells(df,'LogisticRegression',config)
        self.assertEqual(np.isfinite(values).sum(),1)
        self.assertTrue(np.isnan(values).any())
    def test_duplicate_heatmap_cell_rejected(self):
        config=json.loads((ROOT/'config.json').read_text())
        df=pd.DataFrame([{'params_json':'{"model__C":1.0}', 'cv_macro_f1':.9}]*2)
        with self.assertRaises(ValueError):heatmap_cells(df,'LogisticRegression',config)
    def test_layout_detection(self):
        fig,ax=canvas('Valid')
        self.assertEqual(check_layout(fig),[])
        fig.text(1.1,.3,'Outside')
        self.assertTrue(check_layout(fig))
    def test_reordering_preserves_data(self):
        df=pd.DataFrame([{'model':'FisherLDA','method':'pca','macro_f1':.9},
                         {'model':'RBF_SVM','method':'full','macro_f1':.8}])
        sorted_df=_sorted(df,list(MODEL_ORDER))
        self.assertEqual(list(sorted_df.macro_f1),[.8,.9])
    def test_no_report_generator(self):
        self.assertFalse((ROOT/'build_report.py').exists())
        self.assertFalse((ROOT/'src/har_study/reporting.py').exists())

if __name__=='__main__':unittest.main()
