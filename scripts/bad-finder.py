import imageio
import os
import sys
import pickle
import cv2
import tensorflow as tf
tf.enable_eager_execution()
#    try:
#        img_raw = tf.io.read_file(the_record['img_name'])
#        img_load = tf.image.decode_image(img_raw)
#        img_load = tf.image.resize_bilinear(tf.expand_dims(img_load,0), [SIZE, SIZE])
#        img_scaled = tf.cast(img_load, tf.float32)/255.
#        img_final = tf.reshape(img_scaled, [SIZE, SIZE, 3])
#    except Exception as e:
#        print(the_record)
#        print(the_record['img_name'])
#        raise
#    return img_final, the_record['lbl'], the_record['lbp']

def attempt_reshape(imname):
    img_raw = tf.io.read_file(imname)
    img_load = tf.image.decode_image(img_raw)
    img_resize = tf.image.resize_bilinear(tf.expand_dims(img_load, 0), [128,128])
    img_scaled = tf.cast(img_resize,tf.float32)/255.
    img_final = tf.reshape(img_scaled, [128,128,3])

bad_files = []
for imfile in os.listdir(sys.argv[1]):
    try:
        attempt_reshape(os.path.join(sys.argv[1], imfile))
        #im_read = imageio.imread(os.path.join(sys.argv[1], imfile))
        #if len(im_read.shape) != 3:
        #    bad_files.append(imfile)
        #reshape = cv2.resize(im_read, (128, 128), interpolation=cv2.INTER_LINEAR)
    except AttributeError:
        print(imfile)
    except Exception as e:
        print(imfile)
        bad_files.append(imfile)

print(len(bad_files))
with open("bad_file.list", "wb") as p:
    pickle.dump(bad_files, p)
