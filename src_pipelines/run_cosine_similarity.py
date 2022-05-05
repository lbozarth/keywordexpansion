import os
import fasttext
from gensim.models import FastText

PIPELINE_NAME = 'cosine_similarity'

def get_w2v_model(readdir, writedir, file_stem):
    text_filename = os.path.join(readdir, "%s.txt" % file_stem) #file containing tweet text
    model_filename = os.path.join(writedir, "%s_300.bin" % file_stem) #model file
    print(text_filename, model_filename)
    if not os.path.exists(model_filename): #generate model
        print('saving to file', model_filename)
        model = fasttext.train_unsupervised(text_filename, model='skipgram', minCount=50, dim=300, loss='hs')
        model.save_model(model_filename)
    # print('number of words', len(model.words))  # list of words in dictionary
    print('reading model', model_filename)
    fasttext_model = FastText.load_fasttext_format(model_filename, encoding='utf8')
    print('vector for', fasttext_model.most_similar('computer'))
    return

def loop_cs_once(readdir, writedir, file_stem):
    model = get_w2v_model(readdir, writedir, file_stem)
    keywords_inits = [] #input keywords here
    all_top_words = []
    for th in keywords_inits:
        print(th)
        top_words = model.most_similar(th, topn=25)
        top_words = [x[0] for x in top_words]
        all_top_words.extend(top_words)
    all_top_words = list(set(all_top_words))
    print(all_top_words[:10])
    return

if __name__ == '__main__':
    readdir, writedir, file_stem = "", "", "" #TODO add readdir, writedir, and file stem
    loop_cs_once(readdir, writedir, file_stem)