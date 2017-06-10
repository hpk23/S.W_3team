#coding: utf-8

'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import sys
import time
from collections import OrderedDict
import numpy
import six.moves.cPickle as pickle
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import preprocess
import numpy as np

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data): # floatX로 형변환 : GPU환경에서 실행시키기 위함
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False) :
    """
    Used to shuffle the dataset at each iteration. # 반복마다 데이터 세트를 바꿀때 사용
    """
    idx_list = numpy.arange(n, dtype="int32") # 0~n까지의 int형 숫자 생성

    if shuffle:
        numpy.random.shuffle(idx_list) # 리스트를 랜덤으로 섞음

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size) : ## // : 소수점은 버림
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def dropout_layer(state_before, use_noise, trng) : # projection layer : 큰 행렬곱을 피하기 위한 layer
    proj = tensor.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5) 
    # return ndarray or scalar( [0, n]범위의 integers )
    return proj

def get_embedding(embedir):
    f = open(embedir + 'words.lst', "r")
    f2 = open(embedir + "embeddings.txt", "r")
    model = {}

    for line in f:
        line = line.strip() #word
        line2 = f2.readline().strip()
        fields = line2.split()

        data = map(lambda x: float(x), fields)
        model[line] = numpy.array(data)

    return model

def build_pretrained_matrix(Wemd, word_dict, embedding_model):
    num_matched = 0
    num_words = Wemd.shape[0]

    for str, id in word_dict.items():
        if id < num_words and embedding_model.has_key(str) is True:
            Wemd[id] = embedding_model[str]
            num_matched += 1
    return Wemd

def get_layer(name):
    fns = layers[name]
    return fns

def init_params(options, embedding=None) :
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'], options['dim_proj']) # n_words * dim_proj의 랜덤한 행렬
    Wemb = (0.01 * randn).astype(config.floatX) # W embedding

    f = open('vocabulary.pkl', 'rb') # bi lstm
    moph_dict = pickle.load(f) # bi lstm
    if embedding is not None :  # bi lstm
        model = get_embedding('/home/khpark/theano/kor_SRL/glove300d/') # Path
        pp = build_pretrained_matrix(Wemb, moph_dict, model)
        params['Wemb'] = pp
    else :
        params['Wemb'] = Wemb
    
    params = get_layer(options['encoder'])[0](options, params, prefix=options['encoder'])
    # get_layer(options['encoder'])[0] : function >> ** param_init_lstm **, return value : {'lstm_W' : W, 'lstm_U' : U, 'lstm_b' : b}
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'] * 2, options['ydim']).astype(config.floatX)

    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk) # Wemb, lstm_W, lstm_U, lstm_b, U, b에 대한 tparams 리턴
    return tparams

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim) # ndim * ndim 행렬 생성
    u, s, v = numpy.linalg.svd(W) # Wdp e특이값 분해
    return u.astype(config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def param_init_lstm(options, params, prefix='lstm') : 
    """
    Init the LSTM parameter:
    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    # params["lstm_W"] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    # params["lstm_U"] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX) 
    # params["lstm_b"] = b

    return params #{'lstm_W' : W, 'lstm_U' : U, 'lstm_b' : b}

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None) : #이 예제에서는 state_below 에 단어 임베딩값이 들어가게 됨
    nsteps = state_below.shape[0]
    if state_below.ndim == 3 :
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None # mask가 None이 아니면 error를 발생시켜줌

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,                                  # 매 단계의 결과를 list에 담아준다.
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost) :
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost) :
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    """
    Adagrad : Adagrad는 시간에 따른 그래디언트 제곱값을 추적해 학습률을 조정하는 알고리즘입니다.
    잘 업데이트되지 않는 파라메터의 학습률을 높이기 때문에 스파스한 데이터에서 특히 유용하게 쓰입니다.

    Adadelta : Adadelta는 Adagrad를 개선하기 위해 제안된 방법으로,
    하이퍼파라메터에 덜 민감하고 학습률을 너무 빨리 떨어뜨리지 않도록 막습니다.
   """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options) :
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_proj']])
    
    reversed_emb = emb[::-1]  # bi lstm
    reversed_mask = mask[::-1]  # bi lstm

    proj = lstm_layer(tparams, emb, options, mask=mask) # proj는 (n_timesteps x n_samples x dim_proj) 형태의 쌓여진 hidden unit vectors
    reversed_proj = lstm_layer(tparams, reversed_emb, options, mask=reversed_mask) # bi lstm

    proj = tensor.concatenate([proj, reversed_proj], axis=2)  # bi lstm

    if options['encoder'] == 'lstm' :
        proj = (proj * mask[:, :, None]).sum(axis=0)        # mean pooling을 하는 부분
        proj = proj / mask.sum(axis=0)[:, None]

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng) # proj에 dropout을 적용해 다시 return 해줌

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    #maxlen by n_samples by ydim
    #  proj.dot(weight_U) + b 를 한 뒤에 softmax를 적용

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob') # pred_prob를 구하기 위한 function
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')  # pred_prob에서 pred한 class가 어디인지 찾기위한 function

    off = 1e-8
    if pred.dtype == 'float16' :
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False) : # predict 확률
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs

def create_pred(f_pred, prepare_data, data, iterator) :
    out_file = open('predict.out', 'w')
    lines = open('predict.txt', 'r').readlines()
    for _, valid_index in iterator:
        x, mask, y, sentence, title, img = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                            [data[2][t] for t in valid_index],
                                            [data[3][t] for t in valid_index],
                                            [data[4][t] for t in valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        length = len(preds)
        for i in range(length) :
            out_file.write(img[i] + '\t' + title[i] + '\t' + sentence[i] + '\t' + str(preds[i]) + '\n')


def pred_error(f_pred, prepare_data, data, iterator, verbose=False) :
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)

        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def train_lstm(
    dim_proj=300,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.   # Weight decay(=regularization)는 Error에 대한 정의를 수정해서, W의 제곱만큼 다시 빼줌으로써 weight가 너무 빠르게 상승하는 것을 방지
                                                                               # Weight decay 사용시 test set에 대한 성능향상이 많이 일어남
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=10000,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64, # The batch size used for validation/test set.

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=True,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
) :

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = preprocess.load_data, preprocess.prepare_data

    print('Loading data')####################################################
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    ydim = numpy.max(train[1]) + 1   #

    # ydim = 2
    print (ydim)

    model_options['ydim'] = ydim

    print('Building model')####################################################
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, True) # bi-lstm

    if reload_model:
        load_params(saveto, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.

    tparams = init_tparams(params) # # Wemb, lstm_W, lstm_U, lstm_b, U, b에 대한 dict형 tparams 리턴

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))

    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')

    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)
    print('Optimization') 

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf) # calculate err
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or valid_err <= numpy.array(history_errs)[:,0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print('Train Error', train_err, 'Valid Error', valid_err, 'Test Error', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print ( 'Train_Prec ', 1. - train_err, 'Valid_Prec ', 1. - valid_err, 'Test_Prec ', 1. - test_err)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)


def predict_lstm(
    dim_proj=300,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.   # Weight decay(=regularization)는 Error에 대한 정의를 수정해서, W의 제곱만큼 다시 빼줌으로써 weight가 너무 빠르게 상승하는 것을 방지
                                                                               # Weight decay 사용시 test set에 대한 성능향상이 많이 일어남
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='sentiment_model.npz',  # The best model will be saved there
    validFreq=10000,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64, # The batch size used for validation/test set.

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=True,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
) :
    model_options = locals().copy()
    print("model options", model_options)

    predict = preprocess.load_predict_data(n_words=n_words, maxlen=maxlen)

    print('Building model')####################################################

    model_options['ydim'] = 2
    params = init_params(model_options, True) # bi-lstm
    load_params(saveto, params)

    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')
    lr = tensor.scalar(name='lr')

    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)
    print('Optimization')

    kf_predict = get_minibatches_idx(len(predict[0]), valid_batch_size)

    print("%d predict examples" % len(predict[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    try:
        create_pred(f_pred, preprocess.prepare_data, predict, kf_predict)

    except KeyboardInterrupt:
        print("Predict interupted")




if __name__ == '__main__':

    #train_lstm(
    #    saveto='test.npz',
    #    reload_model=True,
    #    validFreq=5000,
    #    saveFreq=5000,
    #)
    predict_lstm()
