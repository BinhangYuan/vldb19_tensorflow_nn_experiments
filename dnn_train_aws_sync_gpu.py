'''
Distributed Tensorflow 0.12.0 example of using data parallelism and share model parameters.

Change the hardcoded host urls below with your own hosts.
Run like this:

pc-01$ python dnn_train_aws_sync_gpu.py --job_name="ps" --task_index=0 --hidden_layer_size=10000
pc-02$ python dnn_train_aws_sync_gpu.py --job_name="worker" --task_index=0 --hidden_layer_size=10000
pc-03$ python dnn_train_aws_sync_gpu.py --job_name="worker" --task_index=1 --hidden_layer_size=10000
pc-04$ python dnn_train_aws_sync_gpu.py --job_name="worker" --task_index=2 --hidden_layer_size=10000
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import time
import subprocess

# cluster specification
parameter_servers = ["100.27.24.224:2222"] # this should be a CPU parameter server.
workers = ["54.210.144.49:2223",
           "100.27.25.125:2223",
           "34.238.154.250:2223"
           ] # these should be GPU workers.
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_boolean("sparse_input", False, "Whether we handle sparse input specially")
tf.app.flags.DEFINE_integer("hidden_layer_size", 10000, "The size of the middle hidden layer")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

# config
num_examples = 4856383
batch_size = 10000//len(workers)
learning_rate = 0.00000001
training_epochs = 1
evaluation_batch_size = 100
logs_path = "/tmp/ffnn/1"
# D1 is number of input features
D1 = 60000
D2 = FLAGS.hidden_layer_size
D3 = D2
C = 17
num_ps_replicas = len(parameter_servers)
num_workers = len(workers)



def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    _, record_string = reader.read(filename_queue)
    record_defaults = [[0.0]] * D1
    record_defaults.append([0])
    record_list = tf.decode_csv(record_string, record_defaults)
    return tf.stack(record_list[0:D1]), tf.one_hot(record_list[D1], C)


def preprocessLR(line):
    splits = line.encode().split('|')
    text = splits[1][1:-1].split(',')
    point = np.empty(len(text))
    for i in range(len(text)):
        if (text[i] is None) or (not text[i].strip()) or (text[i].find('e') != text[i].rfind('e')):
            point[i] = 0.0
        else:
            point[i] = float(text[i])
    return point.astype(np.float32), int(splits[2])


def read_simsql_file_format(filename_queue):
    reader = tf.TextLineReader()
    _, record_string = reader.read(filename_queue)
    record_list = tf.py_func(preprocessLR, [record_string], [tf.float32, tf.int64])
    return tf.reshape(record_list[0], [D1]), tf.one_hot(tf.reshape(record_list[1], []), C)


def preprocess_wiki(line):
    splits = line.decode('utf-8').split(',')
    doc = np.zeros(D1)
    wordcount = int(len(splits) / 2) - 1
    for i in range(wordcount):
        doc[int(splits[2 * i + 2])] = float(splits[2 * i + 3])
    return doc.astype(np.float32), int(splits[1])


def read_wiki_file_format(filename_queue):
    reader = tf.TextLineReader()
    _, record_string = reader.read(filename_queue)
    record_list = tf.py_func(preprocess_wiki, [record_string], [tf.float32, tf.int64])
    return tf.reshape(record_list[0], [D1]), tf.one_hot(tf.reshape(record_list[1], []), C)


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_wiki_file_format(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch




def input_pipeline(filenames, batch_size, num_epochs=None):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    def parse_wiki_format(record_string):
        record_list = tf.py_func(preprocess_wiki, [record_string], [tf.float32, tf.int64])
        return tf.reshape(record_list[0], [D1]), tf.one_hot(tf.reshape(record_list[1], []), C)

    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
            .map(parse_wiki_format)))

    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


def sparse_input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    reader  = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    batch_serialized_examples = tf.train.shuffle_batch(
        [serialized_example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    feature_to_type = {
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'example': tf.SparseFeature(index_key='index', value_key='value', dtype=tf.int64, size=D1, already_sorted=True)
    }
    features = tf.parse_example(batch_serialized_examples, feature_to_type)
    return features['example'], tf.one_hot(features['label'], C, axis=-1)


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        global_step = tf.train.get_or_create_global_step()

        dataset = input_pipeline(['./wiki_data/Wikipedia_tf_60k.csv'], batch_size)
        iterator = dataset.make_one_shot_iterator()
        # input textfiles
        with tf.name_scope('input'):
            x , y_ = iterator.get_next()

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.get_variable("W1", [D1, D2], initializer=tf.random_normal_initializer())
            W2 = tf.get_variable("W2", [D2, D3], initializer=tf.random_normal_initializer())
            W3 = tf.get_variable("W3", [D3, C], initializer=tf.random_normal_initializer())

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([D2]))
            b2 = tf.Variable(tf.zeros([D3]))
            b3 = tf.Variable(tf.zeros([C]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x,W1,a_is_sparse=False),b1)
            a2 = tf.nn.relu(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            a3 = tf.nn.relu(z3)
            z4 = tf.add(tf.matmul(a3,W3),b3)
            y  = tf.nn.log_softmax(z4)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * y, axis=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            hooks = [tf.train.StopAtStepHook(last_step=1000000)]

            grad_op = tf.train.GradientDescentOptimizer(learning_rate)

            rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                    replicas_to_aggregate=len(workers),
                                                    total_num_replicas=len(workers),
                                                    use_locking=True)

            sync_replicas_hook = rep_op.make_session_run_hook(is_chief=(FLAGS.task_index == 0))

            hooks.append(sync_replicas_hook)

            train_op = rep_op.minimize(cross_entropy, global_step=global_step)


        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

        with tf.train.MonitoredTrainingSession(master=server.target,
                            is_chief=(FLAGS.task_index == 0),
                            hooks=hooks) as sess:

            while not sess.should_stop():
                # perform training cycles
                begin_time = time.time()
                frequency = min(1, int(num_examples/batch_size))
                start_time = time.time()

                for epoch in range(training_epochs):
                    # number of batches in one epoch batch_count = int(num_examples/batch_size)
                    batch_count = min(20,int(num_examples/batch_size))
                    count = 0
                    for i in range(batch_count):
                        # perform the operations we defined earlier on batch
                        print("A training iteration begins!")
                        _, cost, summary, step = sess.run([train_op, cross_entropy, summary_op, global_step])
                        print("A trainning iteration ends!")
                        count += 1
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        print("Step: %d," % (step+1),
                              " Epoch: %2d," % (epoch+1),
                              " Batch: %3d of %3d," % (i+1, batch_count),
                              " Cost: %.4f," % cost,
                              " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                        count = 0
                        begin_time = time.time()

            print("done")
