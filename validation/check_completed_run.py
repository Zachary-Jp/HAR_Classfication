"""重新读取原始数据、加载模型复算预测，并比较上传旧运行的共同配置。
不重跑整个网格，也不把旧计时并入新记录。
"""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from har_study.data import load_split,sha256
from har_study.stage_timing import verify_stage_timing
from har_study.verify import verify_run
from har_study.study import dump_json

def main(run):
    run=Path(run)
    base=verify_run(run);stages=verify_stage_timing(run)
    data=load_split(ROOT/'data/UCI HAR Dataset','test')
    saved=pd.read_csv(run/'tables/test_predictions.csv')
    model_rows=[];structures=[]
    with threadpool_limits(limits=1):
        for sid in saved.selection_id.drop_duplicates():
            pipe=joblib.load(run/'models'/f'{sid}.joblib')
            pred=pipe.predict(data.X)
            old=saved[saved.selection_id==sid].sort_values('row_id')
            difference=int(np.count_nonzero(pred!=old.predicted.to_numpy()))
            assert difference==0,(sid,difference)
            model_rows.append({'selection_id':sid,'predictions':len(pred),'mismatches':difference})
            clf=pipe.named_steps['model']
            info={'selection_id':sid,'classifier_input_features':int(clf.n_features_in_)}
            if hasattr(clf,'estimators_'):
                info.update(n_trees=len(clf.estimators_),mean_tree_nodes=float(np.mean([x.tree_.node_count for x in clf.estimators_])),mean_tree_depth=float(np.mean([x.tree_.max_depth for x in clf.estimators_])))
            if hasattr(clf,'support_vectors_'):info['support_vectors']=len(clf.support_vectors_)
            if hasattr(clf,'coef_'):info['coef_shape']=list(clf.coef_.shape)
            structures.append(info)
    record={'status':'PASS','models':model_rows,'total_predictions':sum(r['predictions'] for r in model_rows),
        'total_mismatches':sum(r['mismatches'] for r in model_rows),
        'scope':'重新载入已保存模型并预测；不是再次训练整个参数搜索。'}
    dump_json(ROOT/'validation/reloaded_prediction_check.json',record)
    dump_json(ROOT/'validation/model_structures.json',structures)
    previous=ROOT/'provenance/user_run_snapshot/tables'
    a=pd.read_csv(previous/'cv_folds.csv');b=pd.read_csv(run/'tables/cv_folds.csv')
    keys=['candidate_id','fold'];merged=a.merge(b,on=keys,suffixes=('_previous','_current'),validate='one_to_one')
    diffs={name:float(np.max(np.abs(merged[name+'_previous']-merged[name+'_current']))) for name in ['accuracy','macro_f1','balanced_accuracy'] if name+'_previous' in merged}
    d={'common_candidate_fold_records':len(merged),'max_abs_score_difference':diffs,
        'new_candidate_count':len(pd.read_csv(run/'tables/cv_candidates.csv')),
        'old_candidate_count':len(pd.read_csv(previous/'cv_candidates.csv')),
        'scope':'只比较共同配置的验证指标；新增加250维候选；新计时单独保留。'}
    dump_json(ROOT/'validation/previous_vs_rerun.json',d)
    print(record['total_predictions'],record['total_mismatches'],d)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=ROOT/'outputs/integrated_rerun');args=parser.parse_args();main(args.run)
