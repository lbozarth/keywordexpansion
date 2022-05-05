import ast
import os
import pickle
import socket
import sys
# Disable warning
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src_pipelines import run0_shared

from sklearn.feature_extraction.text import TfidfVectorizer
def basic_vec(key_col, docs, max_df=0.2, min_df=25, max_features=None, stopwords=None, use_idf=True, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, lowercase=False, tokenizer=run0_shared.basic_tokenize, max_df=max_df, min_df=min_df, max_features=max_features,
                          use_idf=use_idf, stop_words=stopwords) #, stop_words=all_stop_words, tokenizer=my_tokenizer

    X = vectorizer.fit_transform(docs)
    count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    print('len_vec', len(key_col.index), len(count_vect_df.index))
    assert(len(key_col.index) == len(count_vect_df.index))
    return vectorizer, count_vect_df

def get_dat(src_fn, wfn, vectorizer=None):
    if not os.path.exists(wfn):
        print('generate features')
        df = pd.read_csv(src_fn, header=0, sep="\t", compression='gzip')
        df.reset_index(drop=True, inplace=True)
        if len(df.index)>200000: #too big
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

def get_clf_dat(base_dir, pipeline):
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

# runs gary king' pipeline (clfs' features generation part)
if __name__ == '__main__':
    pass
