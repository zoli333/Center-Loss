
import tensorflow as tf
import nn



def model_spec(x, keep_prob=0.5, deterministic=False, init=False, use_weight_normalization=False, use_batch_normalization=False, use_mean_only_batch_normalization=False, use_xavier_initialization=False):
    
    # Stage 1
    x = nn.conv2d(x,num_filters=32,filter_size=[5,5],init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv1_1',pad='SAME',nonlinearity=nn.PRelu,use_xavier_initialization=use_xavier_initialization)
    
    x = nn.conv2d(x,num_filters=32,filter_size=[5,5],init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv1_2',pad='SAME',nonlinearity=nn.PRelu,use_xavier_initialization=use_xavier_initialization)
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='max_pool_1_1')
    
    
    # Stage 2
    x = nn.conv2d(x,num_filters=64,filter_size=[5,5],init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv2_1',pad='SAME',nonlinearity=nn.PRelu,use_xavier_initialization=use_xavier_initialization)
    
    x = nn.conv2d(x,num_filters=64,filter_size=[5,5],init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv2_2',pad='SAME',nonlinearity=nn.PRelu,use_xavier_initialization=use_xavier_initialization)
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='max_pool_2_1')
    
    
    
    # Stage 3
    x = nn.conv2d(x,num_filters=128,filter_size=[5,5],init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv3_1',pad='SAME',nonlinearity=nn.PRelu,use_xavier_initialization=use_xavier_initialization)
    
    x = nn.conv2d(x,num_filters=128,filter_size=[5,5],init=init,use_weight_normalization=use_weight_normalization,
                                             use_batch_normalization=use_batch_normalization,
                                             use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                             deterministic=deterministic,
                                             name='conv3_2',pad='SAME',nonlinearity=nn.PRelu,use_xavier_initialization=use_xavier_initialization)
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='max_pool_3_1')
    
    
    
    # embedding layer
    x = tf.reshape(x,[x.get_shape()[0],-1])
    
    embed = nn.dense(x, 2, nonlinearity=None, init=init, 
                                        use_weight_normalization=use_weight_normalization, use_batch_normalization=use_batch_normalization,
                                        use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                        deterministic=deterministic,name='embedding_layer',use_xavier_initialization=use_xavier_initialization)
    
    
    
    x = nn.PRelu(embed, name='embedding_layer/PRelu')
    
    x = nn.dense(x, 10, nonlinearity=None, init=init, 
                                        use_weight_normalization=use_weight_normalization, use_batch_normalization=use_batch_normalization,
                                        use_mean_only_batch_normalization=use_mean_only_batch_normalization,
                                        deterministic=deterministic,name='output_dense',use_xavier_initialization=use_xavier_initialization,use_bias=False)
    
    
    return x, embed
    
    
    
    
    

