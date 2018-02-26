import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from model import model_spec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, dtype=dtypes.uint8)


train_data = np.asarray(mnist.train.images, dtype=np.float32)
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#train_mean = np.mean(train_data)
#train_std = np.std(train_data)
#train_data = (train_data - train_mean) / train_std
test_data = np.asarray(mnist.test.images, np.float32)
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#test_data = (test_data - train_mean) / train_std


train_data = (train_data - 127.5) * 0.0078125
test_data = (test_data - 127.5) * 0.0078125

print train_data.min()
print train_data.max()

print test_data.min()
print test_data.max()

print train_data.std()
print test_data.std()



train_data = train_data.reshape((-1,28,28,1))
test_data = train_data.reshape((-1,28,28,1))


ALPHA = 0.5
LAMBDA = 1.0
num_epochs=50
batch_size=100
num_classes=10
test_batch_size=1000

'''
# plot image
import matplotlib.pyplot as plt
img = np.squeeze(train_data[100])
plt.imshow(img,cmap='gray')
plt.show()
print train_labels[:100]
'''


def center_loss(embedding, labels, num_classes, name=''):
    '''
    embedding dim : (batch_size, num_features)
    '''
    with tf.variable_scope(name):
        
        num_features = embedding.get_shape()[1]
        centroids = tf.get_variable('c',shape=[num_classes,num_features],
                            dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
        centroids_delta = tf.get_variable('centroidsUpdateTempVariable',shape=[num_classes,num_features],dtype=tf.float32,
                            initializer=tf.zeros_initializer(),trainable=False)
        
        
        centroids_batch = tf.gather(centroids,labels)
        
        cLoss = tf.nn.l2_loss(embedding - centroids_batch) / float(batch_size) # Eq. 2
        
        diff = centroids_batch - embedding
        
        delta_c_nominator = tf.scatter_add(centroids_delta, labels, diff)
        
        indices = tf.expand_dims(labels,-1)
        updates = tf.constant(value=1,shape=[indices.get_shape()[0]],dtype=tf.float32)
        shape = tf.constant([num_classes])
        labels_sum = tf.expand_dims(tf.scatter_nd(indices, updates, shape),-1)
        
        
        centroids = centroids.assign_sub(ALPHA * delta_c_nominator / (1.0 + labels_sum))
                
        centroids_delta = centroids_delta.assign(tf.zeros([num_classes,num_features]))
        
        '''
        one_hot_labels = tf.one_hot(y,num_classes)
        labels_sum = tf.expand_dims(tf.reduce_sum(one_hot_labels,reduction_indices=[0]),-1)
        centroids = centroids.assign_sub(ALPHA * delta_c_nominator / (1.0 + labels_sum))
        '''
        return cLoss, centroids
        
        

seed = 3213123
rng = np.random.RandomState(seed)
tf.set_random_seed(seed)



nr_batches_train = int(train_data.shape[0]/batch_size)
nr_batches_test = int(test_data.shape[0]/batch_size)

model = tf.make_template('model',model_spec)


x = tf.placeholder(tf.float32,shape=[batch_size,28,28,1])
y = tf.placeholder(tf.int32,shape=[batch_size])
global_step = tf.Variable(0, trainable=False, name='global_step')


x_test = tf.placeholder(tf.float32,shape=[test_batch_size,28,28,1])
y_test = tf.placeholder(tf.float32,shape=[test_batch_size])

softmax_out, embedding = model(x, use_xavier_initialization=True)

# TODO
center_loss_out, _ = center_loss(embedding, y, num_classes, 'CenterLoss')


softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=softmax_out, name='cross_entropy'))



total_loss = softmax_loss + LAMBDA * center_loss_out

optimizer = tf.train.AdamOptimizer(0.0005).minimize(total_loss,global_step=global_step)

softmax_test_out, embedding_test_out = model(x_test, use_xavier_initialization=True)

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer)
    
    for epoch in range(num_epochs):
        # permute the training data
        inds = rng.permutation(train_data.shape[0])
        trainx_data = train_data[inds]
        trainy = train_labels[inds]
        
        train_err=0.
        for t in range(nr_batches_train):
            feed_dict = {x: train_data[t*batch_size:(t+1)*batch_size], y: train_labels[t*batch_size:(t+1)*batch_size]}
            l,_,g_step=sess.run([total_loss,optimizer,global_step], feed_dict=feed_dict)
            train_err+=l
            if g_step % 200 == 0:
                print l
                labs=train_labels[:test_batch_size]
                feed_dict = {x_test: train_data[:test_batch_size], y_test: labs}
                centroids_to_plot = sess.run(embedding_test_out,feed_dict)
                f = plt.figure(figsize=(16,9))
                c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
                        '#ff00ff', '#990000', '#999900', '#009900', '#009999']
                for i in range(num_classes):
                    plt.plot(centroids_to_plot[labs==i,0].flatten(), centroids_to_plot[labs==i,1].flatten(), '.', c=c[i])
                plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                plt.grid()
                plt.xlim(-2,2)
                plt.ylim(-2,2)
                plt.savefig(str(g_step) + '.png')
                plt.close()
        train_err/=nr_batches_train
        #print train_err
        
        
        
        
        
        
        
        
        
    
    
    
    


