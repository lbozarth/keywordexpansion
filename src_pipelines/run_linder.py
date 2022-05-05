import os
import pickle
import warnings

import pandas as pd
# Disable warning
from sklearn.linear_model import LassoCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PIPELINE_NAME = 'linder'

from src_pipelines import run0_shared

def gridSearchPipeline(X, y, clf, score_fn, parameters):
    grid_search = GridSearchCV(clf, parameters, cv=5, n_jobs=-1, scoring=score_fn) #roc_auc
    grid_search.fit(X, y)
    return grid_search.best_score_, grid_search.best_estimator_.get_params(), grid_search.best_estimator_

def train_clf(X_train, y_train, score_fn='f1'):
    clf_name, clf, parameters = ('SGD', SGDClassifier(), {'alpha':[0.01, 0.001], 'class_weight': ['balanced']})
    print('training', clf_name, clf, parameters)
    score, params, best_clf = gridSearchPipeline(X_train, y_train, clf, score_fn, parameters)
    return clf_name, score, params, best_clf

#f1, precision, recall, roc_auc_score, accuracy_score
def test_clf(X_test, y_test, clf, score_fns=[f1_score, precision_score, recall_score, roc_auc_score, accuracy_score]):
    y_predict = clf.predict(X_test)
    try:
        y_predict_proba = clf.predict_proba(X_test)
        print('y_predict_proba', y_predict_proba[:5])
        y_predict_proba = y_predict_proba[:, 1]
        print('y_predict_proba', y_predict_proba[:5])
    except:
        y_predict_proba = y_predict
    scores = []
    for score_fn in score_fns:
        if 'roc_auc_score' in str(score_fn):
            score = score_fn(y_test, y_predict_proba)
        else:
            score = score_fn(y_test, y_predict)
        scores.append(score)
    return str(scores)

def get_dat(src_fn, wfn, vectorizer=None):
    if not os.path.exists(wfn):
        print('generate features')
        df = pd.read_csv(src_fn, header=0, sep="\t", compression='gzip')
        df.reset_index(drop=True, inplace=True)
        if len(df.index)>200000:
            df = df.groupby('is_protest', as_index=False).apply(lambda x: x.sample(100000))
        df.reset_index(inplace=True)
        if vectorizer is None:
            stopwords = run0_shared.get_stopwords()
            vectorizer, vect_df = run0_shared.basic_vec(df[['id_str']], df['joined_text'].tolist(), max_features=2000, stopwords=stopwords)
            # dump vectorizer
            vect_fn = wfn.replace("train_features.csv.gz", 'vectorizer.pkl')
            with open(vect_fn, 'wb') as f:
                pickle.dump(vectorizer, f)
        else:
            X = vectorizer.transform(df['joined_text'].tolist())
            vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

        df = pd.concat([df[['id_str', 'is_protest', 'joined_text']], vect_df], axis=1)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(wfn, header=True, sep="\t", compression='gzip')
    else:
        df = pd.read_csv(wfn, index_col=0, header=0, sep="\t", compression='gzip')
    return df, vectorizer

def get_clf_dat(base_dir, pipeline=PIPELINE_NAME):
    wfn_train = os.path.join(base_dir, pipeline, 'dat', 'train.csv.gz')
    wfn_test = os.path.join(base_dir, pipeline, 'dat', 'validation.csv.gz')

    wfn_train_feat = os.path.join(base_dir, pipeline, 'dat', 'train_features.csv.gz')
    wfn_test_feat = os.path.join(base_dir, pipeline, 'dat', 'val_features.csv.gz')
    if not os.path.exists(wfn_test_feat):
        print(wfn_train, wfn_train_feat, wfn_test, wfn_test_feat)
        df_train, vectorizer = get_dat(wfn_train, wfn_train_feat)
        df_test, vectorizer = get_dat(wfn_test, wfn_test_feat, vectorizer=vectorizer)
        return df_train, df_test, vectorizer
    else:
        df_train = pd.read_csv(wfn_train_feat, header=0, sep="\t")
        df_test = pd.read_csv(wfn_test_feat, header=0, sep="\t")
        wfn_vec = wfn_train_feat.replace("train_features.csv.gz", 'vectorizer.pkl')
        with open(wfn_vec, 'rb') as f:
            vectorizer = pickle.load(f)
        return df_train, df_test, vectorizer

def run_clf(base_dir):
    clf_name = "SGD"
    wfn = os.path.join(base_dir, PIPELINE_NAME, 'clfs', "%s"%clf_name)
    print('clf file name', wfn)
    df_train, df_test, vectorizer = get_clf_dat(base_dir, PIPELINE_NAME)
    if len(df_train.index)>200000: #data is too big
        df_train = df_train.groupby('is_protest', as_index=False).apply(lambda x: x.sample(100000))
    df_train.reset_index(inplace=True)
    print('train, test shapes', df_train.shape, df_test.shape)

    #training
    features_len = len(vectorizer.get_feature_names())
    X_train, y_train = df_train.iloc[:, -features_len:], df_train['is_protest'].tolist()
    print(X_train.columns)
    clf_name, train_score, params, best_clf = train_clf(X_train, y_train)
    wfn = os.path.join(base_dir, PIPELINE_NAME, 'clfs', "%s.pkl"%clf_name)
    print('writing to file', wfn)
    with open(wfn, 'wb') as f:
        pickle.dump(best_clf, f)

    #testing
    X_test, y_test = df_test.iloc[:, -features_len:], df_test['is_protest'].tolist()
    print(X_test.shape, len(y_test))
    # f1, precision, recall, roc_auc_score, accuracy_score
    test_score = test_clf(X_test, y_test, best_clf)
    print(clf_name, train_score, test_score, params)
    return clf_name, train_score, test_score, params


def run_prediction(base_dir, dat_fn):
    wfn = os.path.join(base_dir, PIPELINE_NAME, 'dat', 'vectorizer.pkl')
    with open(wfn, 'rb') as f:
        vectorizer = pickle.load(f)

    clf_name = 'SGD'
    wfn = os.path.join(base_dir, PIPELINE_NAME, 'clfs', '%s.pkl'%clf_name)
    if not os.path.exists(wfn):
        raise Exception("model is not generated")
    with open(wfn, 'rb') as f:
        best_clf = pickle.load(f)

    write_dir = os.path.join(base_dir, PIPELINE_NAME, 'labeled')
    print(base_dir, write_dir)
    rfn = os.path.join(base_dir, dat_fn)
    df = pd.read_csv(rfn, header=0, sep="\t", compression='gzip')
    df['joined_text'] = df['joined_text'].apply(lambda x: x.split(" "))
    df.reset_index(drop=True, inplace=True)
    # print(df.head())
    X = vectorizer.transform(df['joined_text'].tolist())
    X = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    df = df[['id_str', 'joined_text']]
    y_pred_prob = best_clf.predict(X)
    df['y_pred_prob_%s'%clf_name] = y_pred_prob
    df['y'] = df['y_pred_prob_%s'%clf_name].apply(lambda x: int(x)>0.5)
    df.to_csv(wfn, header=True, index=False, sep="\t", compression='gzip')
    return df

def get_dat_all(base_dir, thread_count=4):
    write_dir = os.path.join(base_dir, PIPELINE_NAME, 'labeled/')
    print(base_dir, write_dir)
    dats = []
    cnt = len(os.listdir(base_dir))
    for i in range(cnt):
        wfn = os.path.join(write_dir, "clean_part%s.csv.gz")
        dats.append(i)
    print('number of chunks', len(dats))
    dats = run0_shared.chunk_bytotal(dats, thread_count)
    return dats

def run_lasso(X, y):
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    feature_name = list(X.columns)
    feature_weights = zip(feature_name, lasso.coef_)
    feature_weights = pd.DataFrame(feature_weights, columns=['keywords', 'lasso_coef'])
    feature_weights = feature_weights[feature_weights['lasso_coef']>0] #positively correlated
    feature_weights.sort_values('lasso_coef', inplace=True, ascending=False)
    return feature_weights

def get_keywords(df, max_df=0.25, min_df=100, max_features=100):
    dfy1 = df[df['y']==1].reset_index(drop=True)
    stopwords = run0_shared.get_stopwords()
    vectorizer, vect_df = run0_shared.basic_vec(dfy1[['id_str']], dfy1['joined_text'], max_df=max_df, min_df=min_df, max_features=max_features, stopwords=stopwords)

    dfy0 = df[df['y']==0].reset_index(drop=True)
    dfy0 = dfy0.sample(len(dfy1.index), replace=False)
    vect_df2 = vectorizer.transform(dfy0['joined_text'])
    vect_df2 = pd.DataFrame(vect_df2.todense(), columns=vectorizer.get_feature_names())

    vect_df = pd.concat([vect_df, vect_df2], axis=0)
    print(vect_df.shape)
    ys = pd.concat([dfy1[['y']], dfy0[['y']]], axis=0)
    keywords = run_lasso(vect_df, ys['y'])
    return keywords

def loop_linder_once():
    print('step 1: generating features')
    base_dir = "" #base data directory
    get_clf_dat(base_dir)

    print('step 2 generating model')
    run_clf(base_dir)

    print('step 3 predicting tweets')
    dat_fn = '' #file containing all the remaining tweets that need prediction
    df_pred = run_prediction(base_dir, dat_fn)

    print('step 4 generate keywords')
    keywords = get_keywords(df_pred)
    print(keywords[:5])
    return

if __name__ == '__main__':
    loop_linder_once()