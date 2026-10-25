from pathlib import Path
import json,hashlib,re
import pandas as pd,numpy as np
P=Path(__file__).resolve().parents[1]
L=['toxic','offensive','hate']; Q=['pred_'+x for x in L]
M=['macro_f1','micro_f1','exact_match']+[x+'_f1' for x in L]
def metrics(d):
 y=d[L].to_numpy(int); z=d[Q].to_numpy(int)
 tp=((y==1)&(z==1)).sum(0); fp=((y==0)&(z==1)).sum(0); fn=((y==1)&(z==0)).sum(0)
 den=2*tp+fp+fn; f=np.divide(2*tp,den,out=np.zeros(3),where=den>0)
 return np.r_[f.mean(),2*tp.sum()/den.sum() if den.sum() else 0,(y==z).all(1).mean(),f]
ref=pd.read_csv(P/'results/classical/oof_LR-TFIDF.csv').sort_values('id').reset_index(drop=True)
checked=0; failures=[]; validation=[]
for family in ['classical','transformers','llm']:
 s=pd.read_csv(P/f'results/{family}/summary.csv').set_index('system')
 for system,row in s.iterrows():
  tag='All-positive' if system=='All-positive (reference)' else system
  files=[P/f'results/{family}/oof_{tag}_seed{v}.csv' for v in [13,21,42,87,100]] if family=='transformers' else [P/f'results/{family}/oof_{tag}.csv']
  runs=[]; pooled=[]
  for f in files:
   d=pd.read_csv(f).sort_values('id').reset_index(drop=True); checked+=1
   assert d[['id','lang','fold']+L].equals(ref[['id','lang','fold']+L]),f
   assert d.id.is_unique and d[Q].isin([0,1]).all().all(),f
   runs.append(np.array([metrics(d[d.fold==k]) for k in range(5)]))
   pooled.append({lang:metrics(d[d.lang==lang])[0] for lang in ['KK','RU','CS']})
  runs=np.array(runs); foldmeans=runs.mean(0)
  for i,m in enumerate(M):
   for suffix,val in [('mean',foldmeans[:,i].mean()),('sd',foldmeans[:,i].std(ddof=1))]:
    if abs(row[f'{m}_{suffix}']-val)>1e-10:failures.append((family,system,m,suffix,float(row[f'{m}_{suffix}']),float(val)))
  for lang in ['KK','RU','CS']:
   val=np.mean([x[lang] for x in pooled])
   if abs(row['macro_f1_'+lang]-val)>1e-10: failures.append((family,system,lang))
  if family=='transformers':
   print(system,'fold/seed mean',round(foldmeans[:,0].mean(),6),'pooled seed-metric mean',round(np.mean([metrics(pd.read_csv(f))[0] for f in files]),6),'pooled ensemble',round(metrics(pd.read_csv(P/f'results/transformers/oof_{tag}_seedmean.csv'))[0],6))
print('Independent metric recomputation: files',checked,'summary errors',len(failures),failures[:5])
print('Language/label totals',ref.groupby('lang')[L].agg(['count','sum']).to_dict())
# Validate each selected LLM prediction came from the validation-selected configuration.
sel=pd.read_csv(P/'results/llm/config_selection.csv')
best=pd.read_csv(P/'results/llm/oof_LLM-best.csv').set_index('id')
for k,g in sel.groupby('fold'):
 chosen=g.loc[g.val_macro_f1.idxmax(),'system']
 assert g.loc[g.chosen,'system'].tolist()==[chosen]
 d=pd.read_csv(P/f'results/llm/oof_{chosen}.csv').set_index('id'); idx=best.index[best.fold==k]
 assert np.array_equal(best.loc[idx,Q],d.loc[idx,Q])
print('LLM configuration selection and chosen predictions: PASS')
assert not failures, failures
# Prediction totals and macro-F1 behind the supplied confusion matrices.
for family,system in [('transformers','XLM-R_seedmean'),('llm','LLM-best')]:
 d=pd.read_csv(P/f'results/{family}/oof_{system}.csv');y=d[L].to_numpy();z=d[Q].to_numpy()
 print('Confusion',system,'TN/FP/FN/TP',np.stack([((y==0)&(z==0)).sum(0),((y==0)&(z==1)).sum(0),((y==1)&(z==0)).sum(0),((y==1)&(z==1)).sum(0)],1).tolist())
