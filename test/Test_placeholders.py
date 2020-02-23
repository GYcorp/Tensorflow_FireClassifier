import tensorflow as tf

batchsize = 64

image_var = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_var = tf.placeholder(tf.float32, [None ,10])

placeholders = [image_var, label_var]

results = [[(elm if elm is not None else batchsize) for elm in ph.get_shape().as_list()] for ph in placeholders]
print(results)

print([ph.get_shape().as_list() for ph in placeholders])

