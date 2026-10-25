from pathlib import Path
import numpy as np,pandas as pd
def precision_score(y,z,**kwargs):
 y,z=np.asarray(y),np.asarray(z);return float(((y==1)&(z==1)).sum()/max((z==1).sum(),1))
def recall_score(y,z,**kwargs):
 y,z=np.asarray(y),np.asarray(z);return float(((y==1)&(z==1)).sum()/max((y==1).sum(),1))
def f1_score(y,z,**kwargs):
 y,z=np.asarray(y),np.asarray(z);return float(2*((y==1)&(z==1)).sum()/max((y==1).sum()+(z==1).sum(),1))
def accuracy_score(y,z):return float((np.asarray(y)==np.asarray(z)).mean())
def roc_auc_score(y,z):
 y,z=np.asarray(y),np.asarray(z);a=z[y==1,None];b=z[y==0][None,:]
 return float(((a>b)+.5*(a==b)).mean())
def average_precision_score(y,z):
 y,z=np.asarray(y),np.asarray(z);order=np.argsort(-z);y,z=y[order],z[order]
 ends=np.r_[np.flatnonzero(np.diff(z)!=0),len(z)-1];tp=np.cumsum(y)[ends]
 return float(np.sum((tp/(ends+1))*np.diff(np.r_[0,tp]))/y.sum())
p=Path(__file__).resolve().parents[1];t=pd.read_csv(p/'results/analysis/label_metrics.csv')
checks=0
for _,r in t.iterrows():
 system,lab=r.system,r.label
 family='transformers' if system in ['XLM-R','mBERT'] else ('llm' if system.startswith('LLM') else 'classical')
 tag='All-positive' if system.startswith('All-positive') else system
 files=[p/f'results/{family}/oof_{tag}_seed{s}.csv' for s in [13,21,42,87,100]] if family=='transformers' else [p/f'results/{family}/oof_{tag}.csv']
 all_runs=[]
 for f in files:
  d=pd.read_csv(f);run=[]
  for k,g in d.groupby('fold'):
   y,z=g[lab],g['pred_'+lab]
   v=[precision_score(y,z,zero_division=0),recall_score(y,z,zero_division=0),f1_score(y,z,zero_division=0),accuracy_score(y,z)]
   if pd.notna(r.roc_auc_mean):v.extend([roc_auc_score(y,g['score_'+lab]),average_precision_score(y,g['score_'+lab])])
   else:v.extend([np.nan,np.nan])
   run.append(v)
  all_runs.append(run)
 a=np.mean(all_runs,axis=0)
 for i,m in enumerate(['precision','recall','f1','accuracy','roc_auc','avg_precision']):
  for suffix,actual in [('mean',a[:,i].mean()),('sd',a[:,i].std(ddof=1))]:
   expected=r[m+'_'+suffix]
   assert (pd.isna(expected) and pd.isna(actual)) or abs(expected-actual)<1e-10,(system,lab,m,suffix,actual,expected)
   checks+=1
print(f'T7 verified: {len(t)} rows, {checks} means/SDs matched independent recomputation.')
