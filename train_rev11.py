import tensorflow as tf
import cv2
import sys
import numpy as np

sess = tf.Session()
sess.run(tf.global_variables_initializer())
count=0

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label


def input_fn(filenames):
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(1024, 1)
  )
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 8)
  )

  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecords"])

def val_input_fn():
    return input_fn(filenames=["val.tfrecords"])

def model_fn(features, labels, mode, params):
    num_classes = 2
    net = features["image"]
    net = tf.identity(net, name="input_tensor")    
    net = tf.reshape(net, [-1, 224, 224, 3])    
    net = tf.identity(net, name="input_tensor_after")    
##==================================================================================
##==================================================================================
    net_conv1 = tf.layers.conv2d(inputs=net, name='layer_conv1',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool1 = tf.layers.max_pooling2d(inputs=net_conv1, pool_size=2, strides=1,padding='same',name='layer1_pool')
##===================================================================================
##===================================================================================    
    net_conv2 = tf.layers.conv2d(inputs=net_pool1, name='layer_conv2',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool2 = tf.layers.max_pooling2d(inputs=net_conv2,pool_size=2, strides=2,padding='same',name='layer2_pool')
##==================================================================================
##==================================================================================    
    net_conv3 = tf.layers.conv2d(inputs=net_pool2, name='layer_conv3',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool3 = tf.layers.max_pooling2d(inputs=net_conv3, pool_size=2, strides=2,padding='same',name='layer3_pool')
    
#====================================================================================
##==================================================================================
    net_conv4 = tf.layers.conv2d(inputs=net_pool3, name='layer_conv4',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool4 = tf.layers.max_pooling2d(inputs=net_conv4, pool_size=2, strides=2,padding='same',name='layer4_pool')
    
#====================================================================================
##==================================================================================
    net_conv5 = tf.layers.conv2d(inputs=net_pool4, name='layer_conv5',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool5 = tf.layers.max_pooling2d(inputs=net_conv5,pool_size=2, strides=2,padding='same',name='layer5_pool')
    
#====================================================================================
##==================================================================================
    net_conv6 = tf.layers.conv2d(inputs=net_pool5, name='layer_conv6',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool6 = tf.layers.max_pooling2d(inputs=net_conv6, pool_size=2, strides=2,padding='same',name='layer6_pool')
    
#====================================================================================
##==================================================================================
    net_conv7 = tf.layers.conv2d(inputs=net_pool6, name='layer_conv7',
                                         filters=64, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
    net_pool7 = tf.layers.max_pooling2d(inputs=net_conv7,pool_size=2, strides=1,padding='same',name='layer7_pool')
    
#====================================================================================
##==================================================================================    
#====================================================================================
    net_flatt = tf.contrib.layers.flatten(net_pool7)
    net = tf.layers.dense(inputs=net_flatt, name='layer_fc1',
                                  units=1024, activation=tf.nn.relu)  
    net_fc1 = tf.layers.dropout(net, rate=0.5, noise_shape=None, 
                                        seed=None, training=(mode == tf.estimator.ModeKeys.TRAIN))
    net_fc2 = tf.layers.dense(inputs=net_fc1, name='layer_fc_2',
                                      units=num_classes)
    logits = net_fc2
    y_pred = tf.nn.softmax(logits=logits)

    y_pred = tf.identity(y_pred, name="output_pred")

    y_pred_cls = tf.argmax(y_pred, axis=1)

    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        metrics={"accuracy": tf.metrics.accuracy(labels, y_pred_cls)}
        spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)

    return spec

model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 1e-3},
                               model_dir="./model/")

learning_count=0
while (count < 100000):
    learning_count=learning_count+2
    rate_of_learning=1e-3/learning_count
    model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": rate_of_learning},
                               model_dir="./model/")
    
    
    
    training_result=model.train(input_fn=train_input_fn, max_steps=None)
    

    result = model.evaluate(input_fn=val_input_fn)
    
    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
    print("learning_rate: ")
    print(rate_of_learning)
    sys.stdout.flush()
    count = count + 1
