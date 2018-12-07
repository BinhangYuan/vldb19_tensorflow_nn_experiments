from __future__ import print_function

import tensorflow as tf
import numpy as np
import time


# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_boolean("sparse_input", False, "Whether we handle sparse input specially")
tf.app.flags.DEFINE_integer("hidden_layer_size", 20000, "The size of the middle hidden layer")
FLAGS = tf.app.flags.FLAGS

# config
num_examples = 4856383
batch_size = 10000
learning_rate = 0.00000001
training_epochs = 1
evaluation_batch_size = 100
logs_path = "./log"
# D1 is number of input features
D1 = 60000
D2 = FLAGS.hidden_layer_size
D3 = D2
C = 17


def preprocess_wiki(line):
    splits = line.decode('utf-8').split(',')
    doc = np.zeros(D1)
    wordcount = int(len(splits) / 2) - 1
    for i in range(wordcount):
        doc[int(splits[2 * i + 2])] = float(splits[2 * i + 3])
    return doc.astype(np.float32), int(splits[1])


def input_pipeline(filenames, batch_size, num_epochs=None):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    def parse_wiki_format(record_string):
        record_list = tf.py_func(preprocess_wiki, [record_string], [tf.float32, tf.int64])
        return tf.reshape(record_list[0], [D1]), tf.one_hot(tf.reshape(record_list[1], []), C)

    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
            .map(parse_wiki_format)))

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset



def wiki_dnn():
    dataset = input_pipeline(['./wiki_data/Wikipedia_tf_60k.csv'], batch_size)
    iterator = dataset.make_one_shot_iterator()
    # input textfiles

    x , y_ = iterator.get_next()

    # model parameters will change during training so we use tf.Variable
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [D1, D2], initializer=tf.random_normal_initializer())
    W2 = tf.get_variable("W2", [D2, D3], initializer=tf.random_normal_initializer())
    W3 = tf.get_variable("W3", [D3, C], initializer=tf.random_normal_initializer())
    b1 = tf.Variable(tf.zeros([D2]))
    b2 = tf.Variable(tf.zeros([D3]))
    b3 = tf.Variable(tf.zeros([C]))

    # implement model
    z2 = tf.add(tf.matmul(x,W1),b1)
    a2 = tf.nn.relu(z2)
    z3 = tf.add(tf.matmul(a2,W2),b2)
    a3 = tf.nn.relu(z3)
    z4 = tf.add(tf.matmul(a3,W3),b3)
    y  = tf.nn.log_softmax(z4)

    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * y, axis=[1]))

    # specify optimizer
    hooks = [tf.train.StopAtStepHook(last_step=100000)]

    global_step = tf.train.get_or_create_global_step()

    rep_op = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = rep_op.minimize(cross_entropy, global_step=global_step)

    # merge all summaries into a single "operation" which we can execute in a session
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    print("Model initialized ...")

    with tf.train.MonitoredTrainingSession(scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op, saver=saver),hooks=hooks) as sess:

        while not sess.should_stop():
            # perform training cycles
            start_time = time.time()

            for epoch in range(training_epochs):
                # number of batches in one epoch batch_count = int(num_examples/batch_size)
                batch_count = 25
                for i in range(batch_count):
                    # perform the operations we defined earlier on batch
                    print("A training iteration begins!")
                    #_, summary, step = sess.run([train_op, summary_op, global_step])
                    _, step = sess.run([train_op, global_step])
                    print("A trainning iteration ends!")
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1),
                            " Batch: %3d of %3d," % (i+1, batch_count),
                            " AvgTime: %3.2fms" % float(elapsed_time*1000))
            print("done")


if __name__ == '__main__':
    wiki_dnn()