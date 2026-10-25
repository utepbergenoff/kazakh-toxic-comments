from pathlib import Path
import subprocess,io
import pandas as pd,numpy as np
P=Path(__file__).resolve().parents[1]
L=['toxic','offensive','hate'];Q=['pred_'+x for x in L]
sample=pd.read_csv(P/'metadata.csv').set_index('id')
ref=pd.read_csv(P/'results/classical/oof_LR-TFIDF.csv').sort_values('id').reset_index(drop=True)
threads=sample.loc[ref.id,'group'].to_numpy()
codes,unique=pd.factorize(threads);G=len(unique)
assert G==545
print('Recovered sample metadata: all evaluated IDs matched;',G,'threads')
check=ref.assign(thread=threads).groupby('thread').fold.nunique()
print('Threads appearing in more than one test fold:',int((check>1).sum()))
W=np.random.default_rng(42).multinomial(G,np.full(G,1/G),size=10000).astype(float)
Y=ref[L].to_numpy(int)
def stats(path):
 d=pd.read_csv(path).sort_values('id').reset_index(drop=True)
 assert d[['id']+L].equals(ref[['id']+L])
 Z=d[Q].to_numpy(int);C=np.zeros((G,3,3))
 for j in range(3):
  for t,mask in enumerate([(Y[:,j]==1)&(Z[:,j]==1),(Y[:,j]==0)&(Z[:,j]==1),(Y[:,j]==1)&(Z[:,j]==0)]):
   np.add.at(C[:,j,t],codes,mask.astype(float))
 def f(c):
  a=c.reshape(-1,3,3);den=2*a[:,:,0]+a[:,:,1]+a[:,:,2]
  return np.divide(2*a[:,:,0],den,out=np.zeros_like(den),where=den>0).mean(1)
 return f(C.sum(0))[0],f(W@C.reshape(G,-1))
S={k:stats(P/v) for k,v in {
 'LR':'results/classical/oof_LR-TFIDF.csv',
 'XLM ensemble':'results/transformers/oof_XLM-R_seedmean.csv',
 'mBERT ensemble':'results/transformers/oof_mBERT_seedmean.csv',
 'LLM':'results/llm/oof_LLM-best.csv',
 'reference':'results/classical/oof_All-positive.csv'}.items()}
x=[stats(P/f'results/transformers/oof_XLM-R_seed{seed}.csv') for seed in [13,21,42,87,100]]
S['XLM seed-metric mean']=(np.mean([v[0] for v in x]),np.mean([v[1] for v in x],axis=0))
y=[stats(P/f'results/transformers/oof_mBERT_seed{seed}.csv') for seed in [13,21,42,87,100]]
S['mBERT seed-metric mean']=(np.mean([v[0] for v in y]),np.mean([v[1] for v in y],axis=0))
# Confirm supplied Figure 2 bootstrap numerical values for all six contrasts.
table=pd.read_csv(P/'results/analysis/bootstrap.csv')
names={'XLM-R':'XLM seed-metric mean','mBERT':'mBERT seed-metric mean','LR-TFIDF':'LR','LLM-best':'LLM','All-positive':'reference'}
for _,r in table.iterrows():
 a,b=S[names[r.system_b]],S[names[r.system_a]]
 obs=[a[0]-b[0],*np.quantile(a[1]-b[1],[.0125,.9875]),np.mean(a[1]-b[1]<=0)]
 assert np.allclose(obs,[r['diff'],r.ci_low,r.ci_high,r.share_diff_le_0],rtol=0,atol=1e-12),r.contrast
print('All six published bootstrap contrasts independently reproduced: PASS')
