import os
import pickle
# Disable warning
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


from src_pipelines import run0_shared, run_king_clfs

PIPELINE_NAME = 'king'

import traceback
def run_prediction(base_dir, read_fn):
    wfn = os.path.join(base_dir, PIPELINE_NAME, 'dat', 'vectorizer.pkl')
    with open(wfn, 'rb') as f:
        vectorizer = pickle.load(f)

    #load clfs
    clfs = []
    for clf_name in run_king_clfs.CLF_NAMES_ALL:
        wfn = os.path.join(base_dir, PIPELINE_NAME, 'clfs', '%s.pkl'%clf_name)
        if not os.path.exists(wfn):
            continue
        with open(wfn, 'rb') as f:
            best_clf = pickle.load(f)
            clfs.append((clf_name, best_clf))
    print('clfs', len(clfs))
    if not clfs:
        return
    # clfs = clfs[:1]

    try:
        rfn = os.path.join(base_dir, read_fn)
        df = pd.read_csv(rfn, header=0, sep="\t", compression='gzip')
        df['joined_text'] = df['joined_text'].apply(lambda x: x.split(" "))
        df.reset_index(drop=True, inplace=True)
        # print(df.head())
        X = vectorizer.transform(df['joined_text'].tolist())
        X = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
        # print(X.shape)
        df = df[['id_str', 'joined_text']]
        for clf_name, best_clf in clfs:
            print('running', clf_name)
            if clf_name == 'SVC' or clf_name=='SGD':
                y_pred_prob = best_clf.predict(X)
            else:
                y_pred_prob = best_clf.predict_proba(X)
                y_pred_prob = y_pred_prob[:, 1]
            df['y_pred_prob_%s'%clf_name] = y_pred_prob

        return df
    except Exception as e:
        print('unexpected', e)
        traceback.print_exc()
    return

def has_one_positive(row):
    for k,v in dict(row).items():
        if "y_pred" in k and v>=0.5:
            return 1
    return 0

def filter_labels(df_all):
    # has at least 1 positive
    print('len0', len(df_all.index))
    df_all['has1'] = df_all.apply(has_one_positive, axis=1)
    df_all = df_all[df_all['has1']==True].reset_index(drop=True)
    print('len1', len(df_all.index))
    return df_all

import hdbscan
from joblib import Memory
def run_clustering_hdbscan(base_dir, df_all):
    max_tweets = 2000000 #2 millions
    if len(df_all.index)>max_tweets:
        df_all = df_all.sample(max_tweets)#sample 25%
    if 'y_pred_prob_SGD' in df_all.columns:
        del df_all['y_pred_prob_SGD']
    X = df_all.iloc[:, 2:]
    print(X.head())
    print('number of data points', X.shape)
    cachedir = os.path.join(base_dir, 'king', 'clusters/mem0')
    print(cachedir)
    mem = Memory(cachedir=cachedir)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=1000, min_samples=100, algorithm='best', memory=mem, core_dist_n_jobs=2)
    clusterer.fit_predict(X)
    df_all['cluster'] = clusterer.labels_
    print(df_all.head(20))
    print(df_all.groupby('cluster').count())

    base_dir = os.path.join(base_dir, 'king', 'labeled')
    wfn_all = os.path.join(base_dir, 'labels_full_results.csv.gz')
    print(wfn_all)
    df_all.to_csv(wfn_all, header=True, index=False, sep="\t")
    return

# runs gary king' pipeline (clfs part)
if __name__ == '__main__':
    pass