import tensorflow as tf

def MixUp_from_tensorflow(x1, p1, alpha):
    beta = tf.distributions.Beta(alpha, alpha).sample(1)[0]
    beta = tf.maximum(beta, 1. - beta)
    
    indexs = tf.random_shuffle(tf.range(tf.shape(x1)[0]))
    x2 = tf.gather(x1, indexs)
    p2 = tf.gather(p1, indexs)
    
    mix_x = beta * x1 + (1 - beta) * x2
    mix_y = beta * p1 + (1 - beta) * p2
    
    return mix_x, mix_y

