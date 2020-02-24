import tensorflow as tf
import numpy as np
import os

if __name__ == '__main__':
    input_vars = tf.placeholder(tf.float32, shape=(None, 60, 60, 4))
    queue = tf.FIFOQueue(1000, [tf.float32], name='queue')
    enqueue_op = queue.enqueue([input_vars])
    output = tf.reduce_mean(queue.dequeue())

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        d = np.random.rand(16, 60, 60, 4)
        for k in range(990):
            enqueue_op.run(feed_dict={input_vars: d})
        # cmd = "ps u " + str(os.getpid()) + " | tail -n 1 | awk '{print $6}'"
        while True:
            # print ("mem in KB: ")
            # os.system(cmd)
            for k in range(300):
                enqueue_op.run(feed_dict={input_vars: d})
                output.eval()