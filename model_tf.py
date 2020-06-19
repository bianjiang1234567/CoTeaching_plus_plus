import numpy as np
import tensorflow as tf
from tensorflow import keras

def compute_pure_ratio(ind1, ind2, indices, noise_or_not ):
    num_remember = len(ind1)
    #print(type(ind1))
    #print(ind2)
    #print(indices)numpy可以索引自己
    '''
    if len(disagree_id)>0:
        pure_ratio_1 = np.sum(noise_or_not[indices[disagree_id[ind1]]]) / float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[indices[disagree_id[ind2]]]) / float(num_remember)
    else:
        pure_ratio_1 = np.sum(noise_or_not[indices[ind1]]) / float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[indices[ind2]]) / float(num_remember)
    '''
    pure_ratio_1 = np.sum(noise_or_not[indices[ind1]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[indices[ind2]]) / float(num_remember)
    return pure_ratio_1, pure_ratio_2


def conv_layer(inputs, filters, kernel_size, strides, padding, training, reuse, name="conv_layer"):
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, epsilon=1e-5, training=training)
        conv = tf.nn.leaky_relu(conv, alpha=0.01)
        return conv


def cnn_model(images, n_outputs, drop_rate, training, top_bn=False, reuse=None, name="cnn_model"):
    # same model as used in the PyTorch version, can be any model theoretically。
    with tf.variable_scope(name, reuse=reuse):
        conv1 = conv_layer(images, filters=128, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_1")
        conv2 = conv_layer(conv1, filters=128, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_2")
        conv3 = conv_layer(conv2, filters=128, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_3")
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)
        #pool3 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
        drop3 = tf.layers.dropout(pool3, rate=drop_rate, training=training,
                                  noise_shape=[tf.shape(pool3)[0], tf.shape(pool3)[1], tf.shape(pool3)[2], 1])

        conv4 = conv_layer(drop3, filters=256, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_4")
        conv5 = conv_layer(conv4, filters=256, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_5")
        conv6 = conv_layer(conv5, filters=256, kernel_size=3, strides=1, padding="same", training=training,
                           reuse=None, name="conv_layer_6")
        pool6 = tf.layers.max_pooling2d(conv6, pool_size=2, strides=2)
        drop6 = tf.layers.dropout(pool6, rate=drop_rate, training=training,
                                  noise_shape=[tf.shape(pool6)[0], tf.shape(pool6)[1], tf.shape(pool6)[2], 1])

        conv7 = conv_layer(drop6, filters=512, kernel_size=3, strides=1, padding="valid", training=training,
                           reuse=None, name="conv_layer_7")
        conv8 = conv_layer(conv7, filters=256, kernel_size=3, strides=1, padding="valid", training=training,
                           reuse=None, name="conv_layer_8")
        conv9 = conv_layer(conv8, filters=128, kernel_size=3, strides=1, padding="valid", training=training,
                           reuse=None, name="conv_layer_9")
        conv9_shape = conv9.get_shape().as_list()
        pool9 = tf.layers.average_pooling2d(conv9, pool_size=conv9_shape[1], strides=conv9_shape[1])

        h = tf.layers.flatten(pool9)
        h = tf.layers.dense(h, units=n_outputs, use_bias=True)
        if top_bn:
            h = tf.layers.batch_normalization(h, momentum=0.9, epsilon=1e-5, training=training)
        logits = h
        predicts = tf.argmax(tf.nn.softmax(h, dim=-1), axis=-1)
        return logits, predicts

def CNN_small(images, n_outputs, reuse=None, name="cnn_model"):#cifar10
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(images, filters=6, kernel_size=5, strides=1, padding="same", name="conv_layer_1", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.he_normal())
        conv = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(conv, pool_size=2, strides=2, padding="valid")
        conv = tf.layers.conv2d(pool, filters=16, kernel_size=5, strides=1, padding="same", name="conv_layer_2", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.he_normal())
        conv = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(conv, pool_size=2, strides=2, padding="valid")
        h = tf.layers.flatten(pool)#fc1 = tf.layers.dense(16 * 5 * 5, 120)
        h = tf.layers.dense(h, units=120, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.he_normal())
        h = tf.layers.dense(h, units=84, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.he_normal())
        h = tf.layers.dense(h, units=n_outputs, use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.he_normal())
        logits = h
        predicts = tf.argmax(tf.nn.softmax(h, dim=-1), axis=-1)
        return logits, predicts


def coteach_loss(logits1, logits2, labels, forget_rate):
    # compute loss
    raw_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels)
    raw_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels)

    # sort and get low loss indices
    #ind1_sorted = tf.argsort(raw_loss1, axis=-1, direction="ASCENDING", stable=True)
    #ind2_sorted = tf.argsort(raw_loss2, axis=-1, direction="ASCENDING", stable=True)
    #num_remember = tf.cast((1.0 - forget_rate) * ind1_sorted.shape[0].value, dtype=tf.int32)
    #num_remember = tf.cast((1.0 - forget_rate) * raw_loss1.shape[0].value, dtype=tf.int32)
    raw_loss1_shape_dynamic = tf.shape(raw_loss1)
    raw_loss1_shape0 = tf.cast(raw_loss1_shape_dynamic[0], dtype=tf.float32)

    #def f1():
    #    return tf.cast((1.0 - forget_rate) * raw_loss1_shape0, dtype=tf.int32)
    #def f2():
    #    return tf.cast(raw_loss1_shape0, dtype=tf.int32)#太少了就不采用smallloss了
    #num_remember = tf.cond(tf.greater((1.0 - forget_rate) * raw_loss1_shape0, 1), fn1=f1, fn2=f2)

    num_remember = tf.cast((1.0 - forget_rate) * raw_loss1_shape0, dtype=tf.int32)

    #if ((1.0 - forget_rate) * raw_loss1_shape0) > 0:
    #    num_remember = tf.cast((1.0 - forget_rate) * raw_loss1_shape0, dtype=tf.int32)
    #else:
    #    num_remember = tf.cast(raw_loss1_shape0, dtype=tf.int32)#防止采用dis的时候样本更少

    #num_remember = 3
    #ind1_update = ind1_sorted[:num_remember]
    #ind2_update = ind2_sorted[:num_remember]
    _ , ind1_update = tf.nn.top_k(input = -raw_loss1, k = num_remember, sorted = True)
    _ , ind2_update = tf.nn.top_k(input = -raw_loss2, k = num_remember, sorted = True)#tensor
    #ind1_update = ind1_update.indices
    #ind2_update = ind2_update.indices

    # update logits and compute loss again
    logits1_update = tf.gather(logits1, ind2_update, axis=0)
    labels1_update = tf.gather(labels, ind2_update, axis=0)
    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1_update, labels=labels1_update)
    loss1 = tf.reduce_sum(loss1) / tf.cast(num_remember, dtype=tf.float32)

    logits2_update = tf.gather(logits2, ind1_update, axis=0)
    labels2_update = tf.gather(labels, ind1_update, axis=0)
    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2_update, labels=labels2_update)
    loss2 = tf.reduce_sum(loss2) / tf.cast(num_remember, dtype=tf.float32)#Tensor 自己运算 算出 Tensor

    disagree_id_tensor = tf.constant([],dtype=tf.int64)
    return loss1, loss2, ind1_update, ind2_update, disagree_id_tensor

def loss_coteaching_plus(logits1, logits2, labels, forget_rate, step):
    #raw_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels)
    #raw_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels)

    #pred1 = tf.argmax(logits1, 1)
    #pred2 = tf.argmax(logits2, 1)
    pred1 = tf.argmax(tf.nn.softmax(logits1, dim=-1), axis=-1)
    pred2 = tf.argmax(tf.nn.softmax(logits2, dim=-1), axis=-1)
    #pred1, pred2 = tf.DType.as_numpy_dtype(pred1), tf.DType.as_numpy_dtype(pred2)

    #logical_disagree_id = np.zeros(labels.shape[0].value, dtype=bool)
    #disagree_id = []

    #NonEqual = tf.not_equal(pred1, pred2)
    NonEqual = tf.equal(pred1, pred2)
    NonEqual_index = tf.where(NonEqual)#二维向量 [ [], [具体坐标从左开始] ]
    NonEqual_index_shape_dynamic = tf.shape(NonEqual_index)

    #NonEqual_index_reshape = tf.reshape(NonEqual_index, [])
    NonEqual_index_reshape = tf.squeeze(NonEqual_index, axis=1)

    #NonEqual_index_one = tf.reduce_any(NonEqual)
    #def f(idx):
    #    disagree_id.append(idx)
    #    logical_disagree_id[idx] = True
    #    return True

    #for idx in range(pred1.shape[0].value):
        #if tf.equal(pred1[idx], pred2[idx]) == tf.constant( False ,dtype = tf.bool):
    #    tf.cond(tf.not_equal(pred1[idx], pred2[idx]), lambda: False, lambda: f(idx))
            #disagree_id.append(idx)
            #logical_disagree_id[idx] = True
    #print(disagree_id)
    #print("len(disagree_id) = ", len(disagree_id))

      #if len(disagree_id) > 0:
    """
    def my_func(x):
        if x:
            return True
        else:
            return False
    flag = tf.py_func(my_func, [NonEqual_index], tf.bool)
    """

    def f1():
        update_labels = tf.gather(labels, NonEqual_index_reshape)
        update_outputs = tf.gather(logits1, NonEqual_index_reshape, axis=0)
        update_outputs2 = tf.gather(logits2, NonEqual_index_reshape, axis=0)
        loss_1, loss_2, ind1_update, ind2_update, _ = coteach_loss(update_outputs, update_outputs2, update_labels,
                                                                   forget_rate)
        # disagree_id = tf.convert_to_tensor_or_sparse_tensor(disagree_id, dtype=tf.int32)
        ind1_update = tf.gather(NonEqual_index_reshape, ind1_update, axis=0)
        ind2_update = tf.gather(NonEqual_index_reshape, ind2_update, axis=0)  # 全是tensor

        ind1_update = tf.cast(ind1_update, dtype=tf.int32)
        ind2_update = tf.cast(ind2_update, dtype=tf.int32)

        return loss_1, loss_2, ind1_update, ind2_update

    def f2():
        _update_step = tf.less(step, tf.constant(5000, dtype = tf.int32))
        update_labels = labels
        update_outputs = logits1
        update_outputs2 = logits2
        loss_1, loss_2, ind1_update, ind2_update, _ = coteach_loss(update_outputs, update_outputs2, update_labels,
                                                                   forget_rate)
        loss_1, loss_2 = tf.cast(_update_step, dtype=tf.float32) * loss_1, tf.cast(_update_step,
                                                                                   dtype=tf.float32) * loss_2

        return loss_1, loss_2, ind1_update, ind2_update

    #aa = tf.cast(NonEqual_index.shape[0], dtype = tf.int32)
    #loss_1, loss_2, ind1_update, ind2_update = tf.cond(tf.equal(tf.reduce_any(tf.not_equal(pred1, pred2)), True), fn1=f1, fn2=f2)
    #loss_1, loss_2, ind1_update, ind2_update = tf.cond(tf.equal(flag, True), fn1=f1, fn2=f2)
    #loss_1, loss_2, ind1_update, ind2_update = tf.cond(tf.greater(NonEqual_index_shape_dynamic[0], 0), fn1=f1, fn2=f2)

    NonEqual_index_shape_dynamic_float =  tf.cast(NonEqual_index_shape_dynamic[0], dtype=tf.float32)
    dis_small_num = (1.0 - forget_rate) * NonEqual_index_shape_dynamic_float
    loss_1, loss_2, ind1_update, ind2_update = tf.cond(tf.greater(dis_small_num, 1.0), fn1=f1, fn2=f2)#保证dis的small loss有

    # disagree_id_tensor = tf.convert_to_tensor(disagree_id, dtype=tf.int32)
    disagree_id_tensor = NonEqual_index_reshape
    # disagree_id_tensor = (NonEqual_index.shape.dims[0] >= tf.Dimension(0))
    return loss_1, loss_2, ind1_update, ind2_update, disagree_id_tensor
"""
    
    #NonEqual_index_array = NonEqual_index.eval()
    #if NonEqual_index.get_shape().as_list()[0] > 0:#####None Values
    #if NonEqual_index.shape.dims[0] > tf.Dimension(0):
    #if NonEqual_index.shape[0].value > 0:不行
    #if 1:
    #if NonEqual_index_shape_dynamic[0] > 0:
    #if NonEqual_index.shape.as_list()[0] > 0:不行
        update_labels = tf.gather(labels, NonEqual_index)
        update_outputs = tf.gather(logits1, NonEqual_index, axis=0)
        update_outputs2 = tf.gather(logits2, NonEqual_index, axis=0)
        loss_1, loss_2, ind1_update, ind2_update, _ = coteach_loss(update_outputs, update_outputs2, update_labels,
                                                                 forget_rate)
        #disagree_id = tf.convert_to_tensor_or_sparse_tensor(disagree_id, dtype=tf.int32)
        ind1_update = tf.gather(NonEqual_index, ind1_update, axis=0)
        ind2_update = tf.gather(NonEqual_index, ind2_update, axis=0)#全是tensor

    else:
        #_update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
        #_update_step = (step < 5000).astype(np.float32)

        _update_step = (tf.less(step, tf.constant(5000, dtype = tf.int32)))
        update_labels = labels
        update_outputs = logits1
        update_outputs2 = logits2
        loss_1, loss_2, ind1_update, ind2_update, _ = coteach_loss(update_outputs, update_outputs2, update_labels,
                                                                forget_rate)
        loss_1, loss_2 =  tf.cast(_update_step, dtype=tf.float32) * loss_1, tf.cast(_update_step, dtype=tf.float32) * loss_2

    #disagree_id_tensor = tf.convert_to_tensor(disagree_id, dtype=tf.int32)
    disagree_id_tensor = NonEqual_index
    #disagree_id_tensor = (NonEqual_index.shape.dims[0] >= tf.Dimension(0))
    return loss_1, loss_2, ind1_update, ind2_update, disagree_id_tensor
"""

class CoTeachingModel:
    def __init__(self, input_shape, n_outputs, mode_type, dataset_name = 'cifar10', batch_size=128, drop_rate=0.25, top_bn=False):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.top_bn = top_bn
        self.mode_type = mode_type
        self._add_placeholder()
        self._build_network(dataset_name)

    def _add_placeholder(self):
        self.images = tf.placeholder(name="images", shape=[self.batch_size] + self.input_shape, dtype=tf.float32)
        self.labels = tf.placeholder(name="labels", shape=[self.batch_size], dtype=tf.int32)
        self.training = tf.placeholder(name="training", shape=[], dtype=tf.bool)#常数没有shape
        self.lr = tf.placeholder(name="learning_rate", shape=[], dtype=tf.float32)
        self.beta1 = tf.placeholder(name="beta1", shape=[], dtype=tf.float32)
        self.forget_rate = tf.placeholder(name="forget_rate", shape=[], dtype=tf.float32)
        self.epoch = tf.placeholder(name="epoch", shape=[], dtype=tf.int32)
        self.init_epoch = tf.placeholder(name="init_epoch", shape=[], dtype=tf.int32)
        self.step = tf.placeholder(name="step", shape=[], dtype=tf.int32)



    def _build_network(self, dataset_name):
        """
                if dataset_name == 'cifar10':
                    logits1, self.predicts1 = CNN_small(self.images, self.n_outputs, name="cnn_model_1")
                    logits2, self.predicts2 = CNN_small(self.images, self.n_outputs, name="cnn_model_2")
                """
        if dataset_name == 'mnist':
            logits1, self.predicts1 = cnn_model(self.images, self.n_outputs, self.drop_rate, self.training, self.top_bn,
                                            reuse=None, name="cnn_model_1")
            logits2, self.predicts2 = cnn_model(self.images, self.n_outputs, self.drop_rate, self.training, self.top_bn,
                                            reuse=None, name="cnn_model_2")
        else:
            logits1, self.predicts1 = CNN_small(self.images, self.n_outputs, name="cnn_model_1")
            logits2, self.predicts2 = CNN_small(self.images, self.n_outputs, name="cnn_model_2")

        self.acc1 = tf.reduce_mean(tf.cast(tf.equal(self.predicts1, tf.cast(self.labels, dtype=tf.int64)),
                                           dtype=tf.float32))
        self.acc2 = tf.reduce_mean(tf.cast(tf.equal(self.predicts2, tf.cast(self.labels, dtype=tf.int64)),
                                           dtype=tf.float32))

        # co-teaching loss
        #if tf.less(self.epoch, self.init_epoch):
        #    self.loss1, self.loss2, self.ind1_update, self.ind2_update = coteach_loss(logits1, logits2, self.labels,
        #                                                                          self.forget_rate)
        #else:
        #    self.loss1, self.loss2, self.ind1_update, self.ind2_update = loss_coteaching_plus(logits1, logits2,
        #                                                                                      self.labels, self.forget_rate, self.step)
        def f1():
            #self.loss1, self.loss2, self.ind1_update, self.ind2_update, self.disagree_id = coteach_loss(logits1, logits2, self.labels, self.forget_rate)
            #return self.loss1, self.loss2, self.ind1_update, self.ind2_update, self.disagree_id
            return coteach_loss(logits1, logits2, self.labels, self.forget_rate)
        def f2():
            #self.loss1, self.loss2, self.ind1_update, self.ind2_update, self.disagree_id = loss_coteaching_plus(logits1, logits2, self.labels, self.forget_rate, self.step)
            #return self.loss1, self.loss2, self.ind1_update, self.ind2_update, self.disagree_id
            return loss_coteaching_plus(logits1, logits2, self.labels, self.forget_rate, self.step)

        if self.mode_type == 'coteaching_plus':
            self.loss1, self.loss2, self.ind1_update, self.ind2_update, self.disagree_id_tensor = tf.cond(self.epoch < self.init_epoch,f1,f2)
        else:
            self.loss1, self.loss2, self.ind1_update, self.ind2_update, self.disagree_id_tensor = coteach_loss(logits1, logits2, self.labels, self.forget_rate)

        # trainable variables
        model1_vars = [x for x in tf.trainable_variables(scope="cnn_model_1")]
        model2_vars = [x for x in tf.trainable_variables(scope="cnn_model_2")]

        # update ops of batch normalization
        self.extra_update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="cnn_model_1")
        self.extra_update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="cnn_model_2")

        # create train operations
        self.train_op1 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.loss1, var_list=model1_vars)
        self.train_op2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.loss2, var_list=model2_vars)


class BaseModel:
    def __init__(self, input_shape, n_outputs, batch_size=128, drop_rate=0.25, top_bn=False):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.top_bn = top_bn
        self._add_placeholder()
        self._build_network()

    def _add_placeholder(self):
        self.images = tf.placeholder(name="images", shape=[self.batch_size] + self.input_shape, dtype=tf.float32)
        self.labels = tf.placeholder(name="labels", shape=[self.batch_size], dtype=tf.int32)
        self.training = tf.placeholder(name="training", shape=[], dtype=tf.bool)
        self.lr = tf.placeholder(name="learning_rate", shape=[], dtype=tf.float32)

    def _build_network(self):
        logits, self.predicts = cnn_model(self.images, self.n_outputs, self.drop_rate, self.training, self.top_bn,
                                          reuse=None, name="cnn_model")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, tf.cast(self.labels, dtype=tf.int64)),
                                               dtype=tf.float32))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        model_vars = [x for x in tf.trainable_variables(scope="cnn_model")]
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=model_vars)
