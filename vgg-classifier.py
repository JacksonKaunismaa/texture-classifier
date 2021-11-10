import tensorflow as tf
from tensorflow.keras.applications import VGG19
import tensorflow.keras.backend as K

SIZE = 224
sess = tf.Session()
K.set_session(sess)

model = VGG19()
model_ins = model.input
model_out = model.layers[15].output
vgg_features = K.function([model_ins, model_out])   # output is (None, 28, 28, 512)










