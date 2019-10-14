from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#=============================================================================

def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def createDataRecord(out_filename, addrs, labels):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        print('Train data: {}/{}'.format(i, len(addrs)))
        sys.stdout.flush()
        img = load_image(addrs[i])

        label = labels[i]
        if img is None:
            continue
#=========================================================================
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()
norms_deffs_train_path = 'new_training/*/*.png'
addrs = glob.glob(norms_deffs_train_path)
labels = [0 if 'one' in addr else 1 for addr in addrs]
c = list(zip(addrs, labels))
shuffle(c)

addrs, labels = zip(*c)
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]

val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

createDataRecord('train.tfrecords', train_addrs, train_labels)
createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('test.tfrecords', test_addrs, test_labels)
