import tensorflow as tf
import numpy as np
import os
import cv2
import sys

META_PATH = "/home/jackson/Code/coding-projects/texture-classifier/models/1.7340210095100648e-23.meta"
MODEL_PATH = "/home/jackson/Code/coding-projects/texture-classifier/models"
translate = {0: "is a texture", 1: "is not a texture"}
def read_image(imname):
    readed = cv2.imread(imname, cv2.IMREAD_COLOR)
    resized = cv2.resize(readed, (64, 64), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized, 0)


with tf.Session() as sess:
    loader = tf.train.import_meta_graph(META_PATH)
    loader.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

    img_input = sess.graph.get_tensor_by_name("main/input:0")
    lbl_outpt = sess.graph.get_tensor_by_name("main/classify:0")

    for img in sys.argv[1:]:
        imload = read_image(img)
        result = int(sess.run(lbl_outpt, feed_dict={img_input: imload})[0][0])
        print(f"{img} {translate[result]}")
