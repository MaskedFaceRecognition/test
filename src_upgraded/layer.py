import tensorflow as tf
# model 학습할 layer

# tf.get_variable : 그것들의 파라미터와 함께 존재하는 variable을 얻거나 새로 variable을 얻는다.
# Computes a 2-D convolution given input and 4-D filters tensors.
def conv_layer(x, filter_shape, stride):
    filters = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        trainable=True)
    return tf.nn.conv2d(input=x, filters=filters, strides=[1, stride, stride, 1], padding='SAME')


def dilated_conv_layer(x, filter_shape, dilation):
    filters = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        trainable=True)
    return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')


def deconv_layer(x, filter_shape, output_shape, stride):
    filters = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        trainable=True)
    return tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1])


def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x=x, axes=[0, 1, 2])
        train_mean = tf.compat.v1.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.compat.v1.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        # Wrapper for Graph.control_dependencies() using the default graph
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.compat.v1.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.compat.v1.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.compat.v1.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.compat.v1.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.constant_initializer(1.0),
        trainable=False)

    return tf.cond(pred=is_training, true_fn=bn_train, false_fn=bn_inference)


def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(a=x, perm=(0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.compat.v1.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    b = tf.compat.v1.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

