"""优化搜索引擎与标准Pipeline的数值等价、人员边界和五模型回归测试。"""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from test_core import fixture, config, small_config, ROOT
sys.path.insert(0, str(ROOT / 'src'))
from har_study.study import MODELS, GRID_KEYS, make_classifier, make_pipeline, grouped_folds, validate_config
from har_study.search import prepare_fold, representation, candidate_grid


class SearchTests(unittest.TestCase):
    def test_five_models_and_candidate_budget(self):
        c = config(); validate_config(c)
        self.assertEqual(len(c['models']), 5)
        self.assertEqual(sum(len(ParameterGrid(c[GRID_KEYS[m]])) for m in c['models']), 32)
        self.assertEqual(32 * (1 + 2 * len(c['dimensions'])), 480)

    def test_new_classifiers(self):
        d=fixture(range(1,4)); c=small_config()
        for name in ('LogisticRegression','KNN','FisherLDA'):
            pipe=make_pipeline(name,'pca',8,c).fit(d.X,d.y)
            self.assertEqual(pipe.predict(d.X).shape, d.y.shape)

    def test_cached_transform_equals_standard_pipeline(self):
        c=small_config(); data=fixture(range(1,13)); folds,_=grouped_folds(data,c)
        tr,va=folds[0]
        with tempfile.TemporaryDirectory() as td, threadpool_limits(limits=1):
            folder=Path(td); prepare_fold(data,tr,va,8,folder)
            for method in ('full','pca','anova'):
                for k in ([24] if method=='full' else [4,8]):
                    X,V,y,truth=representation(folder,method,k)
                    reference=make_pipeline('RBF_SVM',method,k,c)[:-1]
                    expected_X=reference.fit_transform(data.X[tr],data.y[tr])
                    expected_V=reference.transform(data.X[va])
                    np.testing.assert_allclose(X,expected_X,rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(V,expected_V,rtol=1e-10,atol=1e-10)
                    for name in MODELS:
                        cached=make_classifier(name,c).fit(X,y).predict(V)
                        direct=make_pipeline(name,method,k,c).fit(data.X[tr],data.y[tr]).predict(data.X[va])
                        np.testing.assert_array_equal(cached,direct)

    def test_validation_changes_do_not_change_training_transform(self):
        c=small_config(); data=fixture(range(1,13)); tr,va=grouped_folds(data,c)[0][0]
        with tempfile.TemporaryDirectory() as td, threadpool_limits(limits=1):
            a,b=Path(td)/'a',Path(td)/'b'
            prepare_fold(data,tr,va,8,a)
            data.X[va] += 1000
            prepare_fold(data,tr,va,8,b)
            for item in ('train_scaled','train_pca','anova_ranking'):
                np.testing.assert_array_equal(np.load(a/f'{item}.npy'),np.load(b/f'{item}.npy'))

    def test_rejects_fold_leakage(self):
        data=fixture(range(1,4))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): prepare_fold(data,np.arange(20),np.arange(10,30),8,Path(td))


if __name__=='__main__': unittest.main()
