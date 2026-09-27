import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score
def regression_metrics(y_true,y_pred):
    e=np.asarray(y_pred)-np.asarray(y_true); return {"mae":float(np.mean(abs(e))),"rmse":float(np.sqrt(np.mean(e**2)))}
def mard(y_true,y_pred):
    y=np.asarray(y_true); p=np.asarray(y_pred); m=abs(y)>1e-6; return float(np.mean(abs(p[m]-y[m])/abs(y[m]))*100)
def classification_metrics(y_true,prob,threshold=.5):
    y=np.asarray(y_true); p=np.asarray(prob); q=(p>=threshold).astype(int)
    return {"precision":float(precision_score(y,q,zero_division=0)),"recall":float(recall_score(y,q,zero_division=0)),"brier":float(brier_score_loss(y,p)),
            "auroc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan"),"pr_auc":float(average_precision_score(y,p)) if len(np.unique(y))>1 else float("nan")}
def clarke_a_percent(reference,prediction):
    r=np.ravel(reference); p=np.ravel(prediction); a=((r<70)&(p<70))|((r>=70)&(p>=70)&(abs(p-r)/np.maximum(r,1e-6)<=.2))
    return float(100*np.mean(a))
def patient_mean(rows):
    return {k:float(np.nanmean([r[k] for r in rows])) for k in rows[0]} if rows else {}
