import numpy as np
from sklearn.model_selection import GroupKFold
def loso_groups(groups):
    g=np.asarray(groups)
    for p in np.unique(g): yield np.where(g!=p)[0],np.where(g==p)[0]
def group_kfold(groups,n_splits=5):
    g=np.asarray(groups); return GroupKFold(n_splits=n_splits).split(np.zeros(len(g)),groups=g)
def assert_disjoint(train_groups,test_groups):
    overlap=set(train_groups)&set(test_groups)
    if overlap: raise AssertionError(f"patient leakage: {sorted(overlap)}")
