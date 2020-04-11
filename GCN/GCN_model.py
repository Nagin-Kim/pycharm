import tensorflow as tf
import numpy as np

feature=tf.placeholder(tf.float32,shape=(1,))
adjacency=tf.placeholder(tf.float32,shape=(1,))
degree=tf.placeholder(tf.float32,shape=(1,))
label=tf.placeholder(tf.float32,shape=(1,))

weights=tf.Variable(tf.random_normal([,],stddev=1))

def layer(feature,adjacency,degree,weights):
    with tf.name_scope("gcn_layer"):
        d_=tf.pow(degree+tf.eye,-0.5)
        y=tf.matmul(d_,tf.matmul(adjacency,d_))
        kernel=tf.matmul(feature,weights )
        return tf.nn.relu(tf.matmul(y,kernel))

model=layer(feature,adjacency,degree,weights)
with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=label))
    train_op=tf.train.AdamOptimizer(0.001,0.9).minimize(loss)

with tf.Session() as sess:
    sess.run(train_op,feed_dict={feature:feature,adjacency:adjacency,degree=degree,label:label})