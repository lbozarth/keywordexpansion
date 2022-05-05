import os
import pandas as pd

pd.set_option('display.max_rows', 500000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 10000)
pd.set_option('max_colwidth', None)

#################################################################dat
def get_dat_full(base_dir, file_name, chunksize=100000):
    rfn = os.path.join(base_dir, file_name)
    dfs = pd.read_csv(rfn, header=0, sep="\t", chunksize=chunksize, iterator=True, compression='gzip', lineterminator="\n")
    return dfs

import ast
def get_test_dat_clf_full(base_dir, pipeline, split=True):
    rfn = os.path.join(base_dir, pipeline, "dat/traintest.csv.gz")
    df = pd.read_csv(rfn, header=0, sep="\t", compression='gzip')
    if split:
        try:
            df['joined_text'] = df['joined_text'].apply(lambda x: ast.literal_eval(x))
        except:
            df['joined_text'] = df['joined_text'].apply(lambda x: x.split(" "))

    print('total tweets traintest', len(df.index))
    return df

def get_test_dat_clf_train(base_dir, pipeline, split=True):
    rfn = os.path.join(base_dir, pipeline, "dat/train.csv.gz")
    print(rfn)
    df = pd.read_csv(rfn, header=0, sep="\t", compression='gzip')
    if split:
        try:
            df['joined_text'] = df['joined_text'].apply(lambda x: ast.literal_eval(x))
        except:
            df['joined_text'] = df['joined_text'].apply(lambda x: x.split(" "))
    print('total tweets training', len(df.index))
    return df

def get_test_dat_clf_validation(base_dir, pipeline, split=True):
    rfn = os.path.join(base_dir, pipeline, "dat/train.csv.gz")
    print(rfn)
    df = pd.read_csv(rfn, header=0, sep="\t", compression='gzip')
    if split:
        try:
            df['joined_text'] = df['joined_text'].apply(lambda x: ast.literal_eval(x))
        except:
            df['joined_text'] = df['joined_text'].apply(lambda x: x.split(" "))

    print('total tweets validation', len(df.index))
    return df

############################################################### mis
def get_stopwords():
    with open("../data/stopwords.txt", 'r') as f:
        words = f.readlines()
        words = [x.strip() for x in words]
        return words

def chunk_bytotal(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def chunk_bysize(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


#shared resources
if __name__ == '__main__':
    pass