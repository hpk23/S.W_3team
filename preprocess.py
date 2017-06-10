#coding: utf-8

from __future__ import print_function

import gzip
import os

import numpy
import theano
import ast
import pickle

from konlpy.tag import Mecab
import numpy as np


def train_prepare_data(seqs, labels, maxlen=None) :
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels

def prepare_data(seqs, labels, sentences, titles, imgs, maxlen=None) :
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_sentences = []
        new_imgs = []
        new_lengths = []
        for l, s, y, sents, img in zip(lengths, seqs, labels, sentences, imgs):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
                new_sentences.append(sents)
                new_imgs.append(img)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs
        sentences = new_sentences
        imgs = new_imgs

        if len(lengths) < 1:
            return None, None, None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    sentence = []
    title = []
    img = []
    for idx, [s, sent, t, i] in enumerate(zip(seqs, sentences, titles, imgs)):
        sentence.append(sent)
        title.append(t)
        img.append(i)
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels, sentence, title, img


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset) :
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

        
    return dataset


def load_predict_data(path='predict.txt', n_words=100000, maxlen=None,
              sort_by_len=None) :
    unknown_token = u"UNKNOWN_TOKEN"
    mecab = Mecab()
    fv = open('vocabulary.pkl', 'r')
    vocabulary = pickle.load(fv)

    def read_data(filename) :
        with open(filename, 'r') as f :
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[0:]
        return data

    def tokenize(doc) :
        doc = doc.decode('utf-8')
        return ['/'.join(t) for t in mecab.pos(doc)]

    predict_data = read_data(path)
    x_list = []
    y_list = []
    sent_list = []
    title_list = []
    img_list = []

    for i in range(len(predict_data)) :
        sentence = []

        token = tokenize(predict_data[i][1])

        for t in token :
            if t in vocabulary :
                sentence.append(vocabulary[t])
            else :
                sentence.append(vocabulary[unknown_token])


        if(np.sum(sentence) != 0) :
            x_list.append(sentence)
            title_list.append(predict_data[i][0])
            sent_list.append(predict_data[i][1])
            y_list.append(int(predict_data[i][2]))
            img_list.append(predict_data[i][3])

    fv.close()
#//////////////////////////////////////////////////////////////////////////

    predict_set = (x_list, y_list, sent_list, title_list, img_list)

    if maxlen:
        new_predict_set_x = []
        new_predict_set_y = []
        new_predict_set_sent = []
        new_predict_title = []
        new_predict_img = []
        for x, y, sent, title, img in zip(predict_set[0], predict_set[1], predict_set[2], predict_set[3], predict_set[4]) :
            if len(x) < maxlen:
                new_predict_set_x.append(x)
                new_predict_set_y.append(y)
                new_predict_set_sent.append(sent)
                new_predict_title.append(title)
                new_predict_img.append(img)
        predict_set = (new_predict_set_x, new_predict_set_y, new_predict_set_sent, new_predict_title, new_predict_img)
        del new_predict_set_x, new_predict_set_y, new_predict_set_sent, new_predict_title, new_predict_img

    predict_set_x, predict_set_y, predict_set_sent, predict_set_title, predict_set_img = predict_set

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    predict_set_x = remove_unk(predict_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(predict_set_x)
        predict_set_x = [predict_set_x[i] for i in sorted_index]
        predict_set_y = [predict_set_y[i] for i in sorted_index]
        predict_set_sent = [predict_set_sent[i] for i in sorted_index]
        predict_set_title = [predict_set_title[i] for i in sorted_index]
        predict_set_img = [predict_set_img[i] for i in sorted_index]

    predict = (predict_set_x, predict_set_y, predict_set_sent, predict_set_title, predict_set_img)

    return predict




def load_data(n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True) :

    unknown_token = u"UNKNOWN_TOKEN"

    mecab = Mecab()

    fv = open('vocabulary.pkl', 'r')
    vocabulary = pickle.load(fv)

    def read_data(filename) :
        with open(filename, 'r') as f :
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        return data

    def tokenize(doc) :
        doc = doc.decode('utf-8')
        return ['/'.join(t) for t in mecab.pos(doc)]

    # [x][0] : id
    # [x][1] : comment
    # [x][2] : label
    train_data = read_data('ratings_train.txt')
    x_list = []
    y_list = []

    for i in range(len(train_data)) : 
        sentence = []

        token = tokenize(train_data[i][1])

        for t in token :
            if t in vocabulary :
                sentence.append(vocabulary[t])
            else :
                sentence.append(vocabulary[unknown_token])


        if(np.sum(sentence) != 0) :
            x_list.append(sentence)
            y_list.append(int(train_data[i][2]))

    test_data = read_data('ratings_test.txt')
    test_x_list = []
    test_y_list = []

    for i in range(len(test_data)) : 
        sentence = []

        token = tokenize(test_data[i][1])

        for t in token :
            if t in vocabulary :
                sentence.append(vocabulary[t])
            else :
                sentence.append(vocabulary[unknown_token])

        if(np.sum(sentence) != 0) :
            test_x_list.append(sentence)
            test_y_list.append(int(test_data[i][2]))

    fv.close()

    train_set = (x_list, y_list)
    test_set = (test_x_list, test_y_list)

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]) :
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x) # 2103
    sidx = numpy.random.permutation(n_samples) # [0~n_samples) 숫자 random sequence로 list를 생성
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test
