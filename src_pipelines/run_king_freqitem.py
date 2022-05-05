import ast
import os
# Disable warning
import warnings

import nltk.corpus
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
# Disable warning
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src_pipelines import run0_shared

PIPELINE_NAME = 'king'

from sklearn.feature_extraction.text import CountVectorizer
def basic_binary_vec(key_col, docs, max_df=0.5, min_df=0.1, max_features=1000, stopwords=nltk.corpus.stopwords, ngram_range=(1,1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, lowercase=False, tokenizer=run0_shared.basic_tokenize, max_df=max_df, min_df=min_df, max_features=max_features,
                          stop_words=stopwords, binary=True) #, stop_words=all_stop_words, tokenizer=my_tokenizer

    X = vectorizer.fit_transform(docs)
    count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    assert(len(key_col.index) == len(count_vect_df.index))
    return vectorizer, count_vect_df

from scipy.special import gammaln
def compute_bino_likelihood(c1, n1, c2, n2):
    #   c1: count of the event in the first sample.
    #   n1: count of all events for the first sample.
    #   c2: count of the event in the second sample.
    #   n2: count of all events for the second sample.
    p1 = gammaln(c1+1) + gammaln(c2+1) - gammaln(c1+c2+2)
    p2 = gammaln(n1-c1+1) + gammaln(n2-c2+1) - gammaln(n1+n2-c1-c2+2)
    # print(p1, p2)
    return p1 + p2

def gen_item_likelihood(itemset, df):
    df['has_item'] = df['joined_set'].apply(lambda x: int(len(x.intersection(itemset))>0))
    counts = df.groupby('is_cluster').agg({'id_str':'nunique', "has_item":'sum'})
    counts = counts.to_dict("index")
    p = compute_bino_likelihood(counts[1]['has_item'], counts[1]['id_str'], counts[0]['has_item'], counts[0]['id_str'])
    return p

import pickle
def get_word_freq_high(base_dir):
    read_fn = os.path.join(base_dir, 'word_freq_high.pkl')
    with open(read_fn, 'rb') as f:
        fd = pickle.load(f)
        # fd = pd.DataFrame(fd.items(), columns=['word', 'freq'])
    print(fd.head())
    return fd

import traceback
def gen_freqitem(base_dir):
    wfn_all = os.path.join(base_dir, 'labels_full_results.csv.gz')
    print(wfn_all)
    df_all = pd.read_csv(wfn_all, header=0, sep="\t")
    print('total tweets len0', len(df_all.index))
    df_all = df_all[df_all['cluster']!=-1] #not assigned

    # total_size = len(df_all.index)
    # cluster_size = min(1000, (total_size*0.001))
    # print('cluster size is', cluster_size)

    df_all['joined_text'] = df_all['joined_text'].apply(lambda x: ast.literal_eval(x))
    print(df_all.head())
    df_all['cluster_count'] = df_all.groupby('cluster')['id_str'].transform('count')
    df_all = df_all[df_all['cluster_count']>=1000].reset_index(drop=True)
    print('unique clusters', df_all['cluster'].nunique())

    stopwords = run0_shared.get_stopwords()

    df_freq = get_word_freq_high(base_dir)
    freq_words = df_freq['word'].unique()
    print('freq words', list(freq_words)[:10])

    clusters = []
    clusters_v2 = []
    for cluster in df_all['cluster'].unique():
        dfc = df_all[df_all['cluster']==cluster].reset_index(drop=True)
        dfc['is_cluster'] = 1
        try:
            vectorizer, vect_dfc = basic_binary_vec(dfc[['id_str']], dfc['joined_text'].tolist(), stopwords=stopwords)
        except ValueError as e:
            # print(e)  # After pruning, no terms remain. Try a lower min_df or a higher max_df.
            traceback.print_exc()
            continue

        cluster_size = len(dfc.index)
        dfother = df_all[df_all['cluster']!=cluster].reset_index(drop=True)
        if len(dfother.index) > len(dfc.index):
            dfother = dfother.sample(n=len(dfc.index), replace=False)
        dfother['is_cluster'] = 0
        df = pd.concat([dfc, dfother], axis=0)
        df = shuffle(df)
        # print('number of tweets', len(df.index))
        df['joined_set'] = df['joined_text'].apply(lambda x: set(x))
        X = vectorizer.transform(df['joined_text'].to_list())
        vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
        assert(len(df.index)==len(vect_df.index))
        dffp = fpgrowth(vect_df, min_support=0.1, use_colnames=True, max_len=2)
        if not dffp.empty:
            dffp['itemsets_likelihood'] = dffp['itemsets'].apply(gen_item_likelihood, args=(df,))
            dffp.sort_values('itemsets_likelihood', ascending=False, inplace=True)
            #filter out obvious false ones
            #todo get frequency
            # print(dffp.head())

            dffp = dffp.head(10)
            dffp['item'] = dffp['itemsets'].apply(lambda x: "".join(x))
            dffp = dffp[~dffp['item'].isin(freq_words)]
            if not dffp.empty:
                dffp['cluster'] = cluster
                dffp['keywords'] = dffp['itemsets']
                dffp['itemsets'] = dffp.apply(lambda x: "(%s, %s)"%(",".join(x['itemsets']), round(x['support'],2)), axis=1)
                keywords = dffp['itemsets'].tolist()
                clusters.append(dffp)
                clusters_v2.append([cluster, cluster_size, keywords])


    clusters = pd.concat(clusters, axis=0)
    clusters = clusters.groupby(['keywords'], as_index=False).agg({'cluster':'unique', 'support':'mean'})
    clusters['keywords'] = clusters['keywords'].apply(lambda x: " ".join(list(x)))
    print(clusters.head())

    df = pd.DataFrame(clusters_v2, columns=['cluster', 'size', 'keywords'])
    print(df.head())
    return

def test_likelihood():
    result = compute_bino_likelihood(9, 10, 1, 100)
    print(result)
    result = compute_bino_likelihood(1, 10, 1, 100)
    print(result)

    result = compute_bino_likelihood(90, 100, 1, 100)
    print(result)
    result = compute_bino_likelihood(1, 100, 1, 100)
    print(result)

    result = compute_bino_likelihood(50, 100, 50, 100)
    print(result)
    result = compute_bino_likelihood(5, 100, 5, 100)
    print(result)
    return

if __name__ == '__main__':
    # test_likelihood()
    pass