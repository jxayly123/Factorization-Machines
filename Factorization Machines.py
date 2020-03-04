from itertools import count
from collections import defaultdict
from scipy.sparse import csr    
# from __future__ import print_function

def vectorize_dic(dic, ix=None, p=None):
    """ 
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature) 
    
    parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if (ix == None):
        d = count(0)
        ix = defaultdict(lambda: next(d)) 
        
    n = len(list(dic.values())[0]) # num samples
    g = len(list(dic.keys())) # num groups
    nz = n * g # number of non-zeros

    col_ix = np.empty(nz, dtype=int)     
    
    i = 0
    for k, lis in dic.items():     
        # append index el with k in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1
    # print(col_ix, col_ix.shape,col_ix[0])
    row_ix = np.repeat(np.arange(0, n), g)
    # print(col_ix, row_ix)
    data = np.ones(nz)
    # print(data)
    if (p == None):
        p = len(ix)
        
    ixx = np.where(col_ix < p)
    # print(ixx)

    return csr.csr_matrix((data[ixx],(row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix

#load data
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# laod data with pandas
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('./ua.base', delimiter='\t', names=cols)
test = pd.read_csv('./ua.test', delimiter='\t', names=cols)

# vectorize data and convert them to csr matrix
X_train, ix = vectorize_dic({'users': train.user.values, 'items': train.item.values})
X_test, ix = vectorize_dic({'users': test.user.values, 'items': test.item.values}, ix, X_train.shape[1])
y_train = train.rating.values
y_test= test.rating.values

# Densifying the input matrices
X_train = X_train.todense()
X_test = X_test.todense()

# print shape of data
print(X_train.shape)
print(X_test.shape)

# Define FM model with tensorflow
import tensorflow as tf

n, p = X_train.shape

# number of latent factors
k = 10

graph = tf.Graph()
with graph.as_default():
    #design matrix
    with tf.name_scope('train_inputs'):
        X = tf.placeholder('float', shape=[None, p])
        #target vector
        y = tf.placeholder('float', shape=[None, 1])

    with tf.name_scope("weights"):
        # weights
        W = tf.Variable(tf.zeros([p]))

    with tf.name_scope("biases"):
        w0 = tf.Variable(tf.zeros([1]))

    with tf.name_scope("interactions"):
        V = tf.Variable(tf.random_normal([k,p], stddev=0.01))

    with tf.name_scope("y_hat"):
        y_hat = tf.Variable(tf.zeros([n,1]))
        linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
        pair_interactions = (tf.multiply(0.5,
                                         tf.reduce_sum(
                                            tf.subtract(
                                                tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                                                tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                                             1, keep_dims=True
                                         )))
        y_hat = tf.add(linear_terms, pair_interactions)
    with tf.name_scope("loss"):
        lambda_w = tf.constant(0.001, name='lambda_w')
        lambda_v = tf.constant(0.001, name='lambda_v')

        l2_norm = (tf.reduce_sum(
                    tf.add(
                        tf.multiply(lambda_w, tf.pow(W, 2)),
                        tf.multiply(lambda_v, tf.pow(V, 2)))))
        error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
        loss = tf.add(error, l2_norm)

    tf.summary.scalar('loss',loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    merged = tf.summary.merge_all()

    # initialize variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size <1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

# launching tensorflow graph and training the model

from tqdm import tqdm

epochs = 100
batch_size = 1000
log_dir = './train_log'
# Launch the graph
with tf.Session(graph=graph) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(X_train.shape[0])
        #iterate over batches
        for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
            run_metadata = tf.RunMetadata()
            _,summary,t = sess.run([optimizer,merged,loss], feed_dict={X:bX.reshape(-1,p), y:bY.reshape(-1,1)})
            # print(t)
        writer.add_run_metadata(run_metadata, 'epoch{}'.format(epoch))
        writer.add_summary(summary,epoch)
    errors = []
    for bX, bY in batcher(X_test, y_test):
        errors.append(sess.run(error, feed_dict={X: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))

    RMSE = np.sqrt(np.array(errors).mean())
    print(RMSE)
    writer.close()
sess.close()
