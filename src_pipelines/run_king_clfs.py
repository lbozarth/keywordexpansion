import multiprocessing as mp
import os
import pickle
from functools import partial

import pandas as pd
# Disable warning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src_pipelines import  run_king_clfs_features

#(naı̈ve Bayes, k-nearest neighbor(k = 5), logistic regression, support vector machine, decision tree, and random forests)
CLF_NAMES_ALL = ['RandomForest', 'NaiveBayes', 'SVC', 'MLP', 'LogisticRegression', 'DecisionTree', 'SGD'] #'NearestNeighbors'
TOTAL_CLF_COUNT = len(CLF_NAMES_ALL)
DEFAULT_STEM = ""
DEFAULT_CLASSIFIER = [('RandomForest', RandomForestClassifier(), {DEFAULT_STEM + 'max_depth': [4, 5, 6], DEFAULT_STEM + "max_features": ['sqrt', 'log2'], DEFAULT_STEM + 'class_weight': ['balanced']}),
        ('NaiveBayes', GaussianNB(), {}),
        # ('SVC', SVC(), {DEFAULT_STEM+"kernel":['rbf'], DEFAULT_STEM+"gamma":['scale', 'auto'], DEFAULT_STEM + 'class_weight': ['balanced']}),
        # ('SVC', SVC(), {DEFAULT_STEM+"kernel":['linear'], DEFAULT_STEM + 'class_weight': ['balanced']}),
        ('SVC', LinearSVC(dual=False), {DEFAULT_STEM + 'class_weight': ['balanced'], DEFAULT_STEM+'penalty':['l1', 'l2'], DEFAULT_STEM + 'C': [1], DEFAULT_STEM + 'max_iter': [5000]}),
        # ('NearestNeighbors', KNeighborsClassifier(), {DEFAULT_STEM+'n_neighbors':[5], DEFAULT_STEM+'algorithm':['kd_tree']}),
        ('MLP', MLPClassifier(), {DEFAULT_STEM + 'max_iter': [500], DEFAULT_STEM + 'learning_rate_init': [0.001, 0.0001]}), #DEFAULT_STEM + 'alpha': [0.001, 0.01],
        ('LogisticRegression', LogisticRegression(), {DEFAULT_STEM + 'solver': ['liblinear'], DEFAULT_STEM+'penalty':['l1', 'l2'], DEFAULT_STEM + 'class_weight': ['balanced'], DEFAULT_STEM + 'C': [1], DEFAULT_STEM+'max_iter':[5000]}),
        ('DecisionTree', DecisionTreeClassifier(), {DEFAULT_STEM + 'max_depth': [5, 8], DEFAULT_STEM + "max_features": ['sqrt', 'log2'], DEFAULT_STEM + 'class_weight': ['balanced']}),
        ('SGD', SGDClassifier(), {DEFAULT_STEM + 'alpha':[0.01, 0.001], DEFAULT_STEM + 'class_weight': ['balanced']})
        ]
PIPELINE_NAME = 'king'
def gridSearchPipeline(X, y, clf, score_fn, parameters):
    grid_search = GridSearchCV(clf, parameters, cv=5, n_jobs=-1, scoring=score_fn) #roc_auc
    grid_search.fit(X, y)
    # print(str(clf), score_fn, grid_search.best_score_)
    return grid_search.best_score_, grid_search.best_estimator_.get_params(), grid_search.best_estimator_

def train_clf(X_train, y_train, idx, score_fn='f1'):
    clf_name, clf, parameters = DEFAULT_CLASSIFIER[idx]
    print('training', clf_name, clf, parameters)
    score, params, best_clf = gridSearchPipeline(X_train, y_train, clf, score_fn, parameters)
    return clf_name, score, params, best_clf

#f1, precision, recall, roc_auc_score, accuracy_score
def test_clf(X_test, y_test, clf, score_fns=[f1_score, precision_score, recall_score, roc_auc_score, accuracy_score]):
    y_predict = clf.predict(X_test)
    try:
        y_predict_proba = clf.predict_proba(X_test)
        # print('y_predict_proba', y_predict_proba[:5])
        y_predict_proba = y_predict_proba[:, 1]
        # print('y_predict_proba', y_predict_proba[:5])
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

def run_clf(idx):
    clf_name = CLF_NAMES_ALL[idx]
    base_dir = '' #add base_dir
    wfn = os.path.join(base_dir, 'clfs', "%s.pkl" % clf_name)
    print('clf file name', wfn)
    if os.path.exists(wfn):
        return
    df_train, df_test, vectorizer = run_king_clfs_features.get_clf_dat(base_dir, PIPELINE_NAME)
    if len(df_train.index)>100000:
        df_train = df_train.groupby('is_protest', as_index=False).apply(lambda x: x.sample(50000))
    df_train.reset_index(inplace=True)

    if len(df_test.index)>100000:
        df_test = df_test.groupby('is_protest', as_index=False).apply(lambda x: x.sample(50000))
    df_test.reset_index(inplace=True)
    print('train, test shapes', df_train.shape, df_test.shape)

    #training
    features_len = len(vectorizer.get_feature_names())
    X_train, y_train = df_train.iloc[:, -features_len:], df_train['is_protest'].tolist()
    print(X_train.columns)
    clf_name, train_score, params, best_clf = train_clf(X_train, y_train, idx)
    wfn = os.path.join(base_dir, PIPELINE_NAME, 'clfs', "%s.pkl"%clf_name)
    print('writing to file', wfn)
    with open(wfn, 'wb') as f:
        pickle.dump(best_clf, f)

    #testing
    X_test, y_test = df_test.iloc[:, -features_len:], df_test['is_protest'].tolist()
    print('testing shape', X_test.shape, len(y_test))
    # f1, precision, recall, roc_auc_score, accuracy_score
    test_score = test_clf(X_test, y_test, best_clf)
    print(clf_name, train_score, test_score, params)
    return clf_name, train_score, test_score, params


def run_clfs(base_dir, thread_count):
    pool = mp.Pool(processes=thread_count)
    idx_lst = [i for i in range(TOTAL_CLF_COUNT)]
    func = partial(run_clf, )
    ndfs = pool.map(func, idx_lst)
    pool.close()
    pool.join()
    if ndfs:
        results = pd.DataFrame(ndfs, columns = ['clf_name', 'train_score', 'test_scores', 'params'])
        print(results)
        wfn = os.path.join(base_dir, PIPELINE_NAME, 'clfs', "clf_results.csv")
        print(wfn)
        results.to_csv(wfn, header=True, index=False, sep="\t", mode='a')
    return

# runs gary king' pipeline (clfs part)
if __name__ == '__main__':
    # run_clfs(7)
    pass