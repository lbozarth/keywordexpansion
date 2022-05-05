import os
import socket
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Disable warning
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.client import device_lib

from src_pipelines import run0_shared, run_linder

PIPELINE_NAME = 'bert'

def get_test_dat_clf_train(base_dir, split=True):
    df = run0_shared.get_test_dat_clf_train(base_dir, PIPELINE_NAME, split=split)
    if split:
        df['text'] = df['text'].apply(lambda x: x.split(" "))
    return df

def get_test_dat_clf_validation(base_dir, split=True):
    df = run0_shared.get_test_dat_clf_validation(base_dir, PIPELINE_NAME, split=split)
    if split:
        df['text'] = df['text'].apply(lambda x: x.split(" "))
    return df

def gen_model_step2_native(df, tf, model, tokenizer):
    X_train, X_test, y_train, y_test = train_test_split(df['text'].tolist(), df['is_protest'].tolist(),
                                                        stratify=df['is_protest'].tolist(), test_size=0.2)
    print(X_train[:10])
    print(y_train[:10])

    train_encodings = tokenizer(X_train,
                                truncation=True,
                                padding=True)
    # train_dataset = torch.utils.data.TensorDataset(train_encodings, training_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    ))

    val_encodings = tokenizer(X_test,
                              truncation=True,
                              padding=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        y_test
    ))

    model.fit(train_dataset.shuffle(100).batch(16),
              epochs=3,
              batch_size=16,
              validation_data=val_dataset.shuffle(100).batch(16))
    return model

def gen_model(base_dir, model_name):
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    model_fn = os.path.join(base_dir, model_name)
    if os.path.exists(model_fn):
        print("use existing model")
        model = TFDistilBertForSequenceClassification.from_pretrained(model_fn)
    else:
        print('generating model')
        df = get_test_dat_clf_train(split=False)
        #default model
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
        #default tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = gen_model_step2_native(df, tf, model, tokenizer)
        model.save_pretrained(model_fn)
    return model

import traceback
import random
def run_prediction(base_dir, model_name, dat_file):
    # import tensorflow as tf
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    print('running prediction')
    model_fn = os.path.join(base_dir, model_name)
    print('model file', model_fn)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_fn)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    try:
        rfn = os.path.join(base_dir, dat_file)
        df = pd.read_csv(rfn, header=0, sep="\t", compression='gzip')
        df['text'] = df['text'].apply(lambda x: x.lower())
        X = df['text'].tolist()

        predict_input = tokenizer(X,
                                  truncation=True,
                                  padding=True,
                                  return_tensors="tf")

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(predict_input),
        )).batch(16)

        tf_output = model.predict(val_dataset)[0]
        print(tf_output.shape)
        tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()
        print(tf_prediction.shape)
        print(list(tf_prediction)[:10])
        tf_prediction = tf_prediction[:, 1]
        df['y_pred_prob_BERT'] = tf_prediction
        df = df[['id_str', 'joined_text', 'y_pred_prob_BERT']]
        print(df.head(2))
    except Exception as e:
        print(e)
        traceback.print_exc()
    return

def gen_keywords(df):
    features = run_linder.get_keywords(df)
    return features

import time
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    print(names)
    return names

def run_linder_bert_once():
    base_dir, model_name, dat_file = '', '', ''

    print('step 1: generating model')
    gen_model(base_dir, model_name)

    print('step 2 predicting tweets')
    run_prediction(base_dir, model_name, dat_file)

    print('step 3 generate keywords')
    keywords = gen_keywords()
    print(keywords[:5])
    return

if __name__ == '__main__':
    # get_available_gpus()
    run_linder_bert_once()
    pass