import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_random_seed(12)
SIZE = 64
BATCH_SIZE = 32
FILTERS = [3, 512, 256, 128, 64]
KERNEL = 5
FINAL_SIZE = (int(SIZE // (2**(len(FILTERS)-1)))**2) * FILTERS[-1]

LR = 1e-3
WARMUP_EPOCHS = 1
WARMUP_TEMPERATURE = 1.8
EPOCHS = 1000
TRAIN_EPISODES = max(1,40000//BATCH_SIZE)
TEST_EPISODES = max(1,TRAIN_EPISODES//10)


SAVE_PATH = "./models"
LOG_PATH = "./logs"


raw_train = tf.data.TFRecordDataset("./data/train-no-lbp.tfr")
raw_test = tf.data.TFRecordDataset("./data/test-no-lbp.tfr")

tfr_description = {"img_name": tf.FixedLenFeature([], tf.string),
                   "lbl": tf.FixedLenFeature([], tf.int64)}

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
    return img_final, the_record['lbl']

train_data = train_set.map(read_from_dataset)
test_data = test_set.map(read_from_dataset)
train_data = train_data.shuffle(1024).batch(BATCH_SIZE).prefetch(1).repeat()
test_data = test_data.shuffle(1024).batch(BATCH_SIZE).prefetch(1).repeat()
train_iter = train_data.make_one_shot_iterator()
test_iter = train_data.make_one_shot_iterator()
next_tr = train_iter.get_next()
next_te = test_iter.get_next()

def get_weight(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.initializers.glorot_normal(), trainable=True)

def get_bias(name):
    return tf.get_variable(name, shape=[1], initializer=tf.initializers.constant(0.5))

def conv_down(img, conv):
    convd = tf.nn.conv2d(img, conv, [1,2,2,1], padding="SAME")
    actd = tf.nn.relu(convd)
    return actd

def discriminate(img):
    act_img = img
    for i, (in_channels, out_channels) in enumerate(zip(FILTERS, FILTERS[1:])):
        with tf.variable_scope(f"layer-{i}", reuse=tf.AUTO_REUSE):
            conv_mat = get_weight("conv", [KERNEL, KERNEL, in_channels, out_channels])
            act_img = conv_down(act_img, conv_mat)
    with tf.variable_scope("layer-final", reuse=tf.AUTO_REUSE):
        fc = get_weight("fully_connected", [FINAL_SIZE, 1])
        bias = get_bias("bias")
        shaped = tf.reshape(act_img, [-1, FINAL_SIZE])
        logits = tf.reshape(tf.matmul(shaped, fc) + bias, [-1])
    return logits, tf.nn.softmax(logits)

lr_in = tf.placeholder(tf.float32, name="lr")
global_step_tensor = tf.Variable(0, trainable=False, name='gst')

tr_imgs, tr_lbls = next_tr
values, _ = discriminate(tr_imgs)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=values, labels=tf.cast(tr_lbls, tf.float32)))
opt = tf.train.AdamOptimizer(lr_in).minimize(loss, global_step=global_step_tensor)


te_imgs, te_lbls = next_te
te_values, te_probs = discriminate(te_imgs)
with tf.name_scope("test_metrics"):
    te_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=te_values, labels=tf.cast(te_lbls, tf.float32)))
    acc, acc_op = tf.metrics.accuracy(tf.round(te_probs), te_lbls)
    acc_locals = [v for v in tf.local_variables() if "metrics" in v.name]
with tf.variable_scope("layer-1", reuse=True):
    check_me = tf.summary.histogram("conv_hist", get_weight("conv", [KERNEL, KERNEL, FILTERS[1], FILTERS[2]]))
    grads = tf.gradients(loss, get_weight("conv", [KERNEL, KERNEL, FILTERS[1], FILTERS[2]]))[0]

train_summary = tf.summary.merge([tf.summary.scalar("train_loss", loss), check_me])
test_summary  = tf.summary.merge([tf.summary.scalar("test_loss", te_loss), tf.summary.scalar("test_accuracy", acc)])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("Mean grads:", sess.run(grads).mean())
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
        #graph_writer.add_summary(tf.Summary(), 0)
    for j in range(EPOCHS):
        epoch_train_loss = 0.0
        pct_incr = 0.05
        pct = 0.0
        if j < WARMUP_EPOCHS:
            actual_LR = LR*((float(j) / WARMUP_EPOCHS)**WARMUP_TEMPERATURE)
        else:
            actual_LR = LR

        print(f"EPOCH {j} BEGIN: ", flush=True, end='')
        for i in range(TRAIN_EPISODES):
            try:
                train_batch_loss, batch_loss_summ, _ = sess.run([loss, train_summary, opt], feed_dict={lr_in: actual_LR})
                epoch_train_loss += train_batch_loss
                if i % 50 == 0:
                    log_writer.add_summary(batch_loss_summ, get_step())
            except Exception:
                print(j, i)
                raise
            if float(i)/TRAIN_EPISODES >= pct:
                pct += pct_incr
                print("#", end='', flush=True)
        print()
        print(f"Completed training on epoch {j} with average loss per sample of {epoch_train_loss/(TRAIN_EPISODES*BATCH_SIZE)}")
        epoch_test_loss = 0.0
        for i in range(TEST_EPISODES):
            try:
                test_batch_loss, batch_summ, ____ = sess.run([te_loss, test_summary, acc_op])
                log_writer.add_summary(batch_summ, TEST_EPISODES*(get_step()//TRAIN_EPISODES) + i)
                epoch_test_loss += test_batch_loss
                if i % 50 == 0:
                    log_writer.add_summary(batch_summ, (get_step()//TRAIN_EPISODES)*TEST_EPISODES+i)
            except Exception:
                print(j, i)
                raise
        print(f"Completed testing on epoch {j} with average loss per sample of {epoch_test_loss/(TEST_EPISODES*BATCH_SIZE)}")
        print(f"Completed testing on epoch {j} with an accuracy of {sess.run(acc)}")
        sess.run(tf.variables_initializer(acc_locals))  # to reset the accuracy metrics for next epoch
        if j % 5 == 0:
            saver.save(sess, os.path.join(SAVE_PATH, f"model-{get_step()}"))






