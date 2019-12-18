'''
Deep Convolutional Neural Network

Created by @MWransky

'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

image_size = 32
num_channels = 1
batch_size = 64
patch_size = 5
depth = 64
num_hidden = 64

num_labels = 32
reg_parameter = 0.001
learn_rate = 0.01
keep_prob = 0.5


class Network():
    def __init__(self, is_training):
        # Input data.
        self.data = tf.placeholder(shape=[None, image_size, image_size, 1], dtype=tf.float32, name='input')
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32)
        self.label_oh = slim.layers.one_hot_encoding(self.labels, num_labels)

        ### Variables.
        # Weights
        self.layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1),
            name="layer1_weights")
        self.layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1),
            name="layer2_weights")


        self.layer23_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1),
            name="layer2_weights")



        self.layer3_weights = tf.Variable(tf.truncated_normal(
            [16 * image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1),
            name="layer3_weights")
        self.layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1),
            name="layer4_weights")

        # Biases
        # self.layer1_biases = tf.Variable(tf.zeros([depth]), name="layer1_biases")
        # self.layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]),
        #     name="layer2_biases")
        # self.layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]),
        #     name="layer3_biases")
        self.layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]),
            name="layer4_biases")

        # Model
        # 1st Convolution
        self.conv = tf.nn.conv2d(self.data, self.layer1_weights, [1, 1, 1, 1], padding='SAME')
        # print("===========shape: ", self.data.shape, self.layer1_weights.shape)
        # batch normalize conv
        self.conv = batch_norm_wrapper(self.conv, is_training)
        # Relu activation of conv
        self.layer1 = tf.nn.relu(self.conv)
        # Apply dropout
        self.layer1 = tf.nn.dropout(self.layer1, keep_prob)
        
        # 2nd Convolution
        self.conv = tf.nn.conv2d(self.layer1, self.layer2_weights, [1, 1, 1, 1], padding='SAME')
        # batch normalize conv
        self.conv = batch_norm_wrapper(self.conv, is_training)
        # Relu activation of conv
        self.layer2 = tf.nn.relu(self.conv)
        # Apply dropout
        self.layer2 = tf.nn.dropout(self.layer2, keep_prob)


        # 3nd Convolution
        self.conv = tf.nn.conv2d(self.layer2, self.layer23_weights, [1, 1, 1, 1], padding='SAME')
        # batch normalize conv
        self.conv = batch_norm_wrapper(self.conv, is_training)
        # Relu activation of conv
        self.layer23 = tf.nn.relu(self.conv)
        # Apply dropout
        self.layer23 = tf.nn.dropout(self.layer23, keep_prob)
        






        # Resize second layer output for fully connceted layer
        self.shape = self.layer23.get_shape().as_list()
        self.reshape = tf.reshape(self.layer23,
            tf.stack([tf.shape(self.data)[0], self.shape[1] * self.shape[2] * self.shape[3]]))
        # 1st fully connected layer
        self.connected = tf.matmul(self.reshape, self.layer3_weights)
        # batch normalize
        self.connected = batch_norm_wrapper(self.connected, is_training)
        # 1st fully connected layer with relu activation
        self.layer3 = tf.nn.relu(self.connected)
        # Apply dropout
        self.layer3 = tf.nn.dropout(self.layer3, keep_prob)
        # 2nd fully connected layer
        self.logits = tf.matmul(self.layer3, self.layer4_weights) + self.layer4_biases

        #test residual network
        #self.logits = layers.add([self.logits, self.data])

        # Softmax Predictions
        self.probs = tf.nn.softmax(self.logits)

        # Training computation.
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_oh))
        # self.loss = tf.reduce_mean(-tf.reduce_sum(
            # self.label_oh * tf.log(self.probs) + 1e-10, reduction_indices=[1]))

        # Optimizer.
        self.trainer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        # minimization
        self.update = self.trainer.minimize(self.loss)




# Batch Norm Wrapper inspired by http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    sample_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:]), trainable=False)
    sample_variance = tf.Variable(tf.ones(inputs.get_shape()[1:]), trainable=False)

    if is_training:
        batch_mean, batch_variance = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(sample_mean,
            sample_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(sample_variance,
            sample_variance * decay + batch_variance * (1 - decay))
        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_variance, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            sample_mean, sample_variance, beta, scale, epsilon)


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output
