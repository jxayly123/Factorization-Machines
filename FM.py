import numpy as np
# Example dummy data from Rendle 2010
# http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
# Stolen from https://github.com/coreylynch/pyFM
# Categorical variables (Users, Movies, Last Rated) have been one-hot-encoded
x_data = np.matrix([
#    Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
#   A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
])
# ratings
y_data = np.array([5, 3, 1, 4, 5, 1, 5])

# Let's add an axis to make tensoflow happy.
y_data.shape += (1, )

import tensorflow as tf
n, p = x_data.shape
# number of latent factors
k = 5

#design matrix
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("inputs"):
        X = tf.placeholder('float', shape=[n,p])

    with tf.name_scope("outputs"):
        Y = tf.placeholder('float', shape=[n,1])

    with tf.name_scope("weights"):
        w0 = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.zeros([p]))

    with tf.name_scope("interactions"):
        V = tf.Variable(tf.random_normal([k,p], stddev=0.01))

    with tf.name_scope("estimate_y"):
        y_hat = tf.Variable(tf.zeros([n,1]))
        # implement linear regression
        linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X),1, keep_dims=True))
        # implement interaction terms
        interactions = (tf.multiply(0.5,
                                    tf.reduce_sum(
                                        tf.subtract(
                                            tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                                            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V,2)))),
                                        1, keep_dims=True)))
        y_hat = tf.add(linear_terms, interactions)

    # L2 regularized sum of squares loss function over W and V
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
                tf.add(
                    tf.multiply(lambda_w, tf.pow(W,2)),
                    tf.multiply(lambda_v, tf.pow(V,2)))))

    error = tf.reduce_mean(tf.square(tf.subtract(Y, y_hat)))
    with tf.name_scope("loss"):
        loss = tf.add(error, l2_norm)
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()
    eta = tf.constant(0.1)
    optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)
    init = tf.global_variables_initializer()

# iterations
epoches = 1000
log_dir = "./small_train_log"
#launch the graph
with tf.Session(graph=graph) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    for ep in range(epoches):
        run_metadata = tf.RunMetadata()
        indices = np.arange(n)
        np.random.shuffle(indices)
        X_data, Y_data = x_data[indices], y_data[indices]
        _,summary = sess.run([optimizer,merged], feed_dict={X:X_data, Y:Y_data})
        writer.add_run_metadata(run_metadata, 'epoch{}'.format(ep))
        writer.add_summary(summary,ep)
    print('MSE: ', sess.run(error, feed_dict={X: X_data, Y: Y_data}))
    print('Loss (regularized error):', sess.run(loss, feed_dict={X: X_data, Y: Y_data}))
    print('Predictions:', sess.run(y_hat, feed_dict={X: X_data, Y: Y_data}))
    print('Learnt weights:', sess.run(W, feed_dict={X: X_data, Y: Y_data}))
    print('Learnt factors:', sess.run(V, feed_dict={X: X_data, Y: Y_data}))
    writer.close()
sess.close()

