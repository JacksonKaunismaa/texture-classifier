import tensorflow as tf
#import numpy as np
import random
import os
import collections
#import cv2
#from mahotas.features.lbp import lbp
#from multiprocessing import Pool

TEST_AMOUNT = 0.10
LBP_SIZE = 216

#def local_binary(imname):
#    an_im = cv2.imread(imname, cv2.IMREAD_COLOR)
#    np_im = np.array(an_im)
#    result1 = lbp(np_im.max(axis=2), 3, 10)
#    result1 /= result1.sum()
#    result2 = lbp(np_im.mean(axis=2), 2, 10)
#    result2 /= result2.sum()
#    result = np.concatenate((result1, result2))
#    return result

def _floats_feature(val):
    if isinstance(val, collections.Iterable):
        return tf.train.Feature(float_list=tf.train.FloatList(value=val))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))

def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def _int64_feature(val):
    if isinstance(val, collections.Iterable):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def serialize_example_pyf(img_name, lbl):#, bin_pattern):
    feature = {"img_name": _bytes_feature(img_name),
               "lbl": _int64_feature(lbl)}
                #"lbp": _floats_feature(bin_pattern)}
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

def tf_serialize_example(img_name, lbl):#, bin_pattern):
    tf_string = tf.py_func(serialize_example_pyf, (img_name, lbl), tf.string)#, bin_pattern), tf.string)
    return tf.reshape(tf_string, ())

def load_directory(dirname):
    img_files = os.listdir(dirname)
    absolute_img_files = [f"./{dirname}/{x}" for x in img_files][:8000]
    print(f"Found {dirname} dataset with {len(img_files)} images...")
    random.shuffle(absolute_img_files)
    test_size = int(len(img_files)*TEST_AMOUNT)
    test = absolute_img_files[:test_size]
    train = absolute_img_files[test_size:]
    print(f"Created train/test split of size {len(train)}/{len(test)}")
    return train, test

def prep_dataset(full):
    # this means 'texture' is class 1 (lbl=0), "non_texture" is class 2 (lbl=1)
    build = []
    for imname in full:
        if "non_texture/" in imname:
            build.append(1)
        else:
            build.append(0)
    #pool = Pool(8)
    return full, build#, pool.map(local_binary, full)

feature_description = {"img_name": tf.FixedLenFeature([], tf.string),
                       "lbl": tf.FixedLenFeature([], tf.int64)}
                       #"lbp": tf.FixedLenFeature([LBP_SIZE], tf.float32)}
def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    try:
        return tf.parse_single_example(example_proto, feature_description)
    except Exception:
        print(example_proto)
        raise
tf.enable_eager_execution()
train_data1, test_data1 = load_directory("./texture")
train_data2, test_data2 = load_directory("./non_texture")
tr1_tex_size = len(train_data1)
tr2_ntex_size = len(train_data2)
te1_tex_size = len(test_data1)
te2_ntex_size = len(test_data2)
train_data = train_data1 + train_data2
test_data = test_data1 + test_data2
random.shuffle(train_data)
random.shuffle(test_data)
print("Generating local binary pattern (lbp) features for test...")
test_data = prep_dataset(test_data)
print("Generating local binary pattern (lbp) features for train...")
train_data = prep_dataset(train_data)

print("Creating base Datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

print("Mapping Datasets...")
test_dataset = test_dataset.map(tf_serialize_example)
train_dataset = train_dataset.map(tf_serialize_example)
print("Showing example records...")
for final_record in test_dataset.take(5):
    print(_parse_function(final_record))
for final_record in train_dataset.take(5):
    print(_parse_function(final_record))

print("Creating writers...")
train_writer = tf.data.experimental.TFRecordWriter("train.tfr")
test_writer = tf.data.experimental.TFRecordWriter("test.tfr")

print("Writing datasets to tfr files...")
train_writer.write(train_dataset)
test_writer.write(test_dataset)
