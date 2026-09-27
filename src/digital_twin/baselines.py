import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
def persistence_forecast(history,horizon): return np.repeat(np.asarray(history)[:,-1,None],horizon,axis=1)
def linear_trend_forecast(history,horizon):
    x=np.arange(history.shape[1]); xx=np.arange(history.shape[1],history.shape[1]+horizon); return np.asarray([np.polyval(np.polyfit(x,row,1),xx) for row in history])
def static_event_model(X,y): return HistGradientBoostingClassifier(max_iter=100,random_state=42).fit(X,y)
