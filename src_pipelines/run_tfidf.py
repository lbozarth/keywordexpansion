import traceback

import pandas as pd
import socket, sys
import os

PIPELINE_NAME = 'tfidf'

from src_pipelines import run0_shared

from sklearn.feature_extraction.text import TfidfVectorizer
def basic_vec_idf(key_col, docs, max_df=0.25, min_df=100, max_features=None, stopwords=None, use_idf=True, ngram_range=(1,2)):
    # docs = [clean_text(x) for x in docs]
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, lowercase=False, max_df=max_df, min_df=min_df, max_features=max_features,
                          use_idf=use_idf, stop_words=stopwords) #, stop_words=all_stop_words, tokenizer=my_tokenizer

    vectorizer.fit_transform(docs)
    return vectorizer

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

def extract_keywords_row(row, vectorizer):
    # sort the tf-idf vector by descending order of scores
    feature_names = vectorizer.get_feature_names()
    try:
        vec = vectorizer.transform([row['joined_text']])
        sorted_items = sort_coo(vec.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 3)
        return list(keywords.items())
    except:
        traceback.print_exc()

def gen_keywords(df):
    stopwords = run0_shared.get_stopwords()
    vectorizer = basic_vec_idf(df[['id_str']], df['joined_text'].tolist(), min_df=0.1, max_features=300, stopwords=stopwords)

    df['keywords'] = df.apply(extract_keywords_row, args=(vectorizer, ), axis=1)
    df = df.explode('keywords')
    df = df[~df['keywords'].isnull()]
    df['keyword'] = df['keywords'].apply(lambda x: x[0])
    df['tfidf_weight'] = df['keywords'].apply(lambda x: x[1])
    df = df[['keyword', 'tfidf_weight']]
    df = df.groupby(['keyword'], as_index=False).agg({'tfidf_weight':'mean'})
    df.sort_values('tfidf_weight', inplace=True, ascending=False)
    return df['keyword'].tolist()

def loop_tfidf_once():
    df = run0_shared.get_test_dat_clf_full(PIPELINE_NAME)
    df = df[df['is_protest']==1].reset_index(drop=True)
    print('len', len(df.index))
    keywords = gen_keywords(df)
    print(keywords[:10])
    return

if __name__ == '__main__':
    loop_tfidf_once()