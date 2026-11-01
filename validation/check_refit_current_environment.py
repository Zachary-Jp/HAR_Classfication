"""复拟合锁定配置；只检查同环境复现，不替换正式结果或计时。"""
from pathlib import Path
import json,sys,time,tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from har_study.data import load_split
from har_study.study import make_pipeline,grouped_folds,dump_json,metrics
from har_study.search import prepare_fold,score_fold
run=ROOT/'outputs/integrated_rerun';c=json.loads((run/'config_frozen.json').read_text())
train=load_split(ROOT/'data/UCI HAR Dataset','train');test=load_split(ROOT/'data/UCI HAR Dataset','test')
sels=pd.read_csv(run/'tables/selected_configs.csv');preds=pd.read_csv(run/'tables/test_predictions.csv')
rows=[]
with threadpool_limits(limits=1):
 for row in sels.to_dict('records'):
  p=make_pipeline(row['model'],row['method'],int(row['k']),c)
  p.set_params(**json.loads(row['params_json']));t=time.perf_counter();p.fit(train.X,train.y)
  pred=p.predict(test.X);old=preds[preds.selection_id==row['selection_id']].sort_values('row_id')
  n=int(np.sum(pred!=old.predicted.to_numpy()))
  rows.append({'selection_id':row['selection_id'],'n':len(pred),'mismatches':n,'elapsed_seconds':time.perf_counter()-t})
  print(rows[-1],flush=True)
 r={'status':'PASS' if all(x['mismatches']==0 for x in rows) else 'DIFFERENCES_FOUND','records':rows,
  'total_predictions':sum(x['n'] for x in rows),'total_mismatches':sum(x['mismatches'] for x in rows),
  'scope':'同环境重新拟合15个锁定Pipeline；非整个2400次搜索重跑。未覆盖正式模型和计时。'}
 dump_json(ROOT/'validation/current_environment_refit.json',r)
 # 单独核查与旧包差异最大的LR候选，排查本环境缓存复用是否改变结果。
 tab=pd.read_csv(run/'tables/cv_candidates.csv');folds,_=grouped_folds(train,c)
 spec=tab[(tab.model=='LogisticRegression')&(tab.method=='anova')&(tab.k==300)].to_dict('records')
 spec=next(x for x in spec if json.loads(x['params_json'])['model__C']==100)
 tr,va=folds[0];result={}
 with tempfile.TemporaryDirectory() as td:
  prepare_fold(train,tr,va,max(c['dimensions']),Path(td))
  cached=score_fold(spec,0,Path(td),c)
  ordinary=make_pipeline('LogisticRegression','anova',300,c).set_params(**json.loads(spec['params_json']))
  ordinary.fit(train.X[tr],train.y[tr]);p=ordinary.predict(train.X[va]);ordinaryscore=metrics(train.y[va],p)
  original=pd.read_csv(run/'tables/cv_folds.csv');target=original[(original.candidate_id==spec['candidate_id'])&(original.fold==0)].iloc[0]
  result={'candidate_id':spec['candidate_id'],'fold':0,'stored_current_f1':float(target.macro_f1),
   'repeat_cached_f1':cached['macro_f1'],'ordinary_pipeline_f1':ordinaryscore['macro_f1'],
   'note':'原包与新环境间的LR差异原因尚不能仅由这些记录确定。当前环境结果分别如上，不隐去差异。'}
 dump_json(ROOT/'validation/lr_diagnostic.json',result);print(result,flush=True)
