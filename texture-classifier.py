import tensorflow as tf
import os
#from tensorflow.keras.applications import VGG19
#import tensorflow.keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_random_seed(12)
SIZE = 128
BATCH_SIZE = 32
#LBP_SIZE = 216
FILTERS = [3, 512, 256, 128, 64]
#HIDDEN = [LBP_SIZE, 64, 32, 1]
KERNEL = 8
FINAL_SIZE = int(SIZE // (2**(len(FILTERS)-1)))

LR = 1e-3
WARMUP_EPOCHS = 0
WARMUP_TEMPERATURE = 1.8
EPOCHS = 1000
TRAIN_EPISODES = max(1,10//BATCH_SIZE)
TEST_EPISODES = max(1,TRAIN_EPISODES//10)
PKEEP = 0.8  # means 40% of all logits will be 0


SAVE_PATH = "./models"
LOG_PATH = "./logs"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#K.set_session(sess)
#model = VGG19()

#assert len(HIDDEN) == len(FILTERS), "incorrect layer configuration (num fully_connected != num convolutional)"
raw_train = tf.data.TFRecordDataset("./data/train-no-lbp.tfr")
raw_test = tf.data.TFRecordDataset("./data/test-no-lbp.tfr")

tfr_description = {"img_name": tf.FixedLenFeature([], tf.string),
                   "lbl": tf.FixedLenFeature([], tf.int64)}
#                   "lbp": tf.FixedLenFeature([LBP_SIZE], tf.float32)}

def _parse_tfr(proto):
    return tf.parse_single_example(proto, tfr_description)

train_set = raw_train.map(_parse_tfr)
test_set = raw_test.map(_parse_tfr)

def read_from_dataset(the_record):
    try:
        img_raw = tf.io.read_file(the_record['img_name'])
        img_load = tf.image.decode_image(img_raw)
        img_resized = tf.image.resize_bilinear(tf.expand_dims(img_load,0), [SIZE, SIZE])
        img_scaled = tf.cast(img_resized, tf.float32)/255.
        img_final = tf.reshape(img_scaled, [SIZE, SIZE, 3])
    except Exception:
        print(the_record)
        print(the_record['img_name'])
        raise
    return img_final, the_record['lbl']#, the_record['lbp']*1e10

train_data = train_set.map(read_from_dataset)
test_data = test_set.map(read_from_dataset)
train_data = train_data.shuffle(1024).batch(BATCH_SIZE).prefetch(1).repeat()
test_data = test_data.shuffle(1024).batch(BATCH_SIZE).prefetch(1).repeat()
train_iter = train_data.make_one_shot_iterator()
test_iter = train_data.make_one_shot_iterator()
next_tr = train_iter.get_next()
next_te = test_iter.get_next()


def get_theta(name, shape):
    theta = tf.get_variable(name, shape=shape, trainable=True,
                            initializer=tf.random_normal_initializer(stddev=0.4))
    return theta

def get_bias(name, shape):
    bias = tf.get_variable(name, shape=shape, trainable=True,
                           initializer=tf.constant_initializer(0.5))
    return bias

def circular_pad(z, amount):
    z = tf.concat(   #! first part pads the top with last few, second part pads the bottom
        (z[:, -amount:, :], z, z[:, :amount, :]),
        axis=1)

    z = tf.concat(   #! same but left, then right gets padded
        (z[:, :, -amount:], z, z[:, :, :amount]),
        axis=2)
    return z

def convolve_down(input_tensor, convolver):
#    input_tensor = circular_pad(input_tensor, (tf.shape(convolver)[0]-1)//2)
    down_conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, 2, 2, 1], padding="SAME")
    down_conv_a = tf.nn.relu(down_conv)
    return down_conv_a


def fully_connected(input_tensor, weight_matrix, bias, activation):
    logits = tf.matmul(input_tensor, weight_matrix) + bias
    #batch_norm = tf.layers.batch_normalization(logits, axis=-1, training=train_mode,
    #                                           scale=False, center=False)
    non_linear = activation(logits)
    #dropped = tf.nn.dropout(non_linear, pkeep)
    #return dropped
    return non_linear

def discriminate(img_input):#, lbp_input):
    #act_lbp = tf.reshape(lbp_input, [-1, LBP_SIZE])
    act_img = img_input
#    with tf.variable_scope("frequency", reuse=tf.AUTO_REUSE):
#        transposed = tf.transpose(img_input, [0, 3, 1, 2])    # turn [N, H, W, C] -> [N, C, H, W] (compute fft per batch and channel)
#        dfft_img_complex = tf.spectral.rfft2d(transposed, fft_length=[128,128])
#        fft_img = tf.transpose(tf.cast(dfft_img_complex, tf.float32), [0, 2, 3, 1])[:,:,1:,:]   # turn [N, C, H, W] -> [N, H, W, C]

    for idx, (in_channels, out_channels) in enumerate(zip(FILTERS, FILTERS[1:])):#, HIDDEN, HIDDEN[1:])):
        with tf.variable_scope(f"layer-{idx}", reuse=tf.AUTO_REUSE):
            conv = get_theta("conv", [KERNEL, KERNEL, in_channels, out_channels])
            #fc = get_theta("fc", [in_hidden, out_hidden])
            #bias = get_bias("bias", [out_hidden])
#            fft_conv = get_theta("fft_conv", [KERNEL, KERNEL, in_channels, out_channels])

#            fft_img = convolve_down(fft_img, fft_conv)
            act_img = convolve_down(act_img, conv)
            #act_lbp = fully_connected(act_lbp, fc, bias, tf.nn.tanh)

    with tf.variable_scope("final-layer-img", reuse=tf.AUTO_REUSE):
        act_img_final = tf.reshape(act_img, [-1, FINAL_SIZE*FINAL_SIZE*FILTERS[-1]])
        final_img_fc = get_theta("fc", [FINAL_SIZE*FINAL_SIZE*FILTERS[-1], 1])
        final_img_bias = get_bias("bias", [1])
        final_img = fully_connected(act_img_final, final_img_fc, final_img_bias, tf.nn.relu)

#    with tf.variable_scope("final-layer-fft", reuse=tf.AUTO_REUSE):
#        act_fft_final = tf.reshape(fft_img, [-1, FINAL_SIZE*FINAL_SIZE*FILTERS[-1]//2])
#        final_fft_fc = get_theta("fft_fc", [FINAL_SIZE*FINAL_SIZE*FILTERS[-1]//2, 1])
#        final_fft_bias = get_bias("fft_bias", [1])
#        final_fft = fully_connected(act_fft_final, final_fft_fc, final_fft_bias, tf.tanh)

    with tf.variable_scope("final-layer", reuse=tf.AUTO_REUSE):
    #    img_weight = get_theta("img_weight", [1])
        #lbp_weight = get_theta("lbp_weight", [1])
#       fft_weight = get_theta("fft_weight", [1])
        final_logits = final_img#*img_weight# + lbp_weight*act_lbp# + fft_weight*final_fft
        return tf.nn.sigmoid(final_logits, name="classifier"), act_img, final_img

with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
    train_mode = tf.placeholder(tf.bool, name='train_mode')
    pkeep = tf.placeholder(tf.float32, name='pkeep')
    learn_rate = tf.placeholder(tf.float32, name="learn_rate")
    #production_img = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3], name='prod_img')
    #production_lbp = tf.placeholder(tf.float32, [None, LBP_SIZE], name='prod_lbp')
    #classified = tf.identity(discriminate(production_img, production_lbp), name="classify")
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    imgs, lbls = next_tr#, lbps = next_tr
    train_out, conv_out, fc_conv_out = discriminate(imgs)#, lbps)
    loss = tf.reduce_mean(tf.square(train_out - tf.cast(lbls, tf.float32)))
    opt = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss, global_step=global_step_tensor)

    te_imgs, te_lbls = next_te#, te_lbps = next_te
    test_out,____,________ = discriminate(te_imgs)#, te_lbps)
    test_loss = tf.reduce_mean(tf.square(test_out - tf.cast(te_lbls, tf.float32)))
    correct_predictions = tf.equal(tf.cast(tf.round(test_out), tf.int64), te_lbls)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

weight_summary = []#[tf.summary.scalar("img_weight", get_theta("final-layer/img_weight", [1])[0])]
with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
    weight_summary.append(tf.summary.histogram("conv_weight", get_theta("layer-0/conv", [KERNEL, KERNEL, FILTERS[0], FILTERS[1]])))
                  #tf.summary.scalar("lbp_weight", get_theta("final-layer/img_weight", [1])[0]),
                  #tf.summary.scalar("fft_weight", get_theta("final-layer/fft_weight", [1])[0])]
test_summary = [tf.summary.scalar("accuracy", accuracy),
                tf.summary.scalar("test_loss", test_loss)]
train_summary = [tf.summary.scalar("train_loss", loss)]

weight_summary_op = tf.summary.merge(weight_summary)
test_summary_op = tf.summary.merge(test_summary)
train_summary_op = tf.summary.merge(train_summary)

sess.run(tf.global_variables_initializer())

with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
    gr = tf.gradients(loss, get_theta("layer-3/conv", [KERNEL, KERNEL, FILTERS[3], FILTERS[4]]))
    #gr = tf.gradients(loss, get_theta("final-layer/img_weight", [1]))
gr_res = sess.run(gr, feed_dict={pkeep:PKEEP, train_mode:True})[0]
print(gr_res)
print("mean", gr_res.mean())
import numpy as np
from PIL import Image

avg_tex = 0.0
avg_ntex = 0.0
cnt_tex = 0.0
cnt_ntex = 0.0
for _ in range(100):
    im_batch, lbl_batch = sess.run([imgs,lbls])
    for img_training,lbl_rn in zip(im_batch, lbl_batch):
        if lbl_rn == 0:
            avg_tex += img_training.mean()
            cnt_tex += 1.0
        else:
            avg_ntex += img_training.mean()
            cnt_ntex += 1.0
print(avg_tex / cnt_tex)
print(avg_ntex / cnt_ntex)
def show(imdat):
    pil_im = Image.fromarray((255.*imdat).astype(np.uint8), "RGB")
    pil_im.show()
loss_batch, im_batch, lbl_batch, res_batch, conv_pred, conv_fc = sess.run([loss, imgs, lbls, train_out, conv_out, fc_conv_out], feed_dict={train_mode:True, pkeep:PKEEP})
print("LOSS", loss_batch)
for i, img_train in enumerate(im_batch):
    print("LABEL", lbl_batch[i])
    print("NETWORK", res_batch[i])
    print("CONV_PRED: AVG, SHAPE", conv_pred[i].mean(), conv_pred[i].shape)
    print("CONV_FC_PRED", conv_fc[i])
    show(img_train)
    input()

while True:
    input()

quit()

def get_step():
    return tf.train.global_step(sess, global_step_tensor)
saver = tf.train.Saver()
loader = tf.train.Saver()
log_writer = tf.summary.FileWriter(LOG_PATH)
try:
    loader.restore(sess, tf.train.latest_checkpoint(SAVE_PATH))
except ValueError:
    print("No models found, initializing random model...")
    graph_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    graph_writer.add_summary(tf.Summary(), 0)
for j in range(EPOCHS):
    epoch_train_loss = 0.0
    pct_incr = 0.05
    pct = 0.0
    if j < WARMUP_EPOCHS:
        actual_LR = LR*((float(j)/WARMUP_EPOCHS)**WARMUP_TEMPERATURE)
    else:
        actual_LR = LR
    print(f"EPOCH {j} BEGIN: ", flush=True, end='')
    for i in range(TRAIN_EPISODES):
        try:
            train_batch_loss, batch_loss_summ, _ = sess.run([loss, train_summary_op, opt], feed_dict={learn_rate: actual_LR, pkeep:PKEEP, train_mode:True})
            log_writer.add_summary(batch_loss_summ, get_step())
            epoch_train_loss += train_batch_loss
        except Exception:
            print(j, i)
            raise
        if float(i)/TRAIN_EPISODES >= pct:
            pct += pct_incr
            print("#", end='', flush=True)
    print(flush=True)
    print(f"Completed training on epoch {j} with average loss per sample of {epoch_train_loss/(TRAIN_EPISODES*BATCH_SIZE)}")
    weighting_summary = sess.run(weight_summary_op)
    log_writer.add_summary(weighting_summary, get_step())
    epoch_test_loss = 0.0
    for i in range(TEST_EPISODES):
        try:
            test_batch_loss, batch_summ = sess.run([test_loss, test_summary_op], feed_dict={pkeep:1.0, train_mode:False})
            log_writer.add_summary(batch_summ, TEST_EPISODES*(get_step()//TRAIN_EPISODES) + i)
            epoch_test_loss += test_batch_loss
        except Exception:
            print(j, i)
    print(f"Completed testing on epoch {j} with average loss per sample of {epoch_test_loss/(TEST_EPISODES*BATCH_SIZE)}")
    if j % 5 == 0:
        saver.save(sess, os.path.join(SAVE_PATH, f"model-{get_step()}"))
sess.close()
