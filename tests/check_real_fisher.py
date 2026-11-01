"""可选数值核验：仅使用官方训练人员的一折，不是完整搜索或最终测试。

运行：python tests/check_real_fisher.py
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from har_study.data import find_dataset_root,load_split
from har_study.study import grouped_folds,make_classifier,make_pipeline
from har_study.search import prepare_fold,representation


def check():
    config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    train=load_split(find_dataset_root(ROOT/'data'),'train')
    tr,va=grouped_folds(train,config)[0][0]
    records=[]
    with tempfile.TemporaryDirectory(prefix='har_fisher_check_') as td,threadpool_limits(limits=1):
        cache=Path(td)
        prepare_fold(train,tr,va,200,cache)
        for method in ('full','pca','anova'):
            k=561 if method=='full' else 200
            X,V,y,truth=representation(cache,method,k)
            for shrinkage in config['lda_grid']['model__shrinkage']:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model=make_classifier('FisherLDA',config).set_params(shrinkage=shrinkage)
                    model.fit(X,y);a=model.predict(V)
                    pipeline=make_pipeline('FisherLDA',method,k,config).set_params(model__shrinkage=shrinkage)
                    pipeline.fit(train.X[tr],train.y[tr]);b=pipeline.predict(train.X[va])
                mismatches=int(np.sum(a!=b))
                if mismatches:raise AssertionError(f'{method}/{shrinkage}预测不同：{mismatches}')
                records.append({'method':method,'k':k,'shrinkage':shrinkage,'validation_rows':len(va),
                                'prediction_mismatches':mismatches,
                                'warnings':[str(w.message) for w in caught]})
    return {'status':'PASS','scope':'训练人员第一折；共享预处理与完整Pipeline的FisherLDA预测一致性，不是新的测试集实验成绩。',
            'train_subjects':sorted(map(int,np.unique(train.subjects[tr]))),
            'validation_subjects':sorted(map(int,np.unique(train.subjects[va]))),
            'n_configurations':len(records),'prediction_pairs_compared':sum(r['validation_rows'] for r in records),
            'cases':records}

if __name__=='__main__':
    result=check()
    target=ROOT/'validation/fisher_training_fold_check.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"[Check] {result['n_configurations']} Fisher configurations; {result['prediction_pairs_compared']} prediction pairs; 0 differences.")
