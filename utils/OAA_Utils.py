import tensorflow as tf
import os
import cv2
import numpy as np

OAA_save_index = 0

def Online_Attention_Accumulation(attention_maps, probs, save_paths, labels, last_epoch=False):
    """accumulate attentions

    Args:
        attention maps: ndarray of attention maps (before normalized) [b, w, h, c].
        probs: list of class probability (after sigmoid).
        labels: list of class labels (one hot vector).
        save paths: list of images name(or path).
        last epoch: boolean to force to save attention map.

    Returns:
        returns nothing

    """
    atts = attention_maps * (attention_maps > 0) # relu

    for image_index in range(len(attention_maps)):
        # loop only existing label
        for label_index in np.nonzero(labels[image_index])[0]:

            att = atts[image_index,:,:,label_index]
            prob = probs[image_index,label_index]
            save_path = save_paths[image_index].replace('.jpg','_{}.jpg'.format(label_index))

            # make dir if not exits
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            # normalize 0~1
            att = att / (att.max() + 1e-8) * 255

            # if this is last epoch and the image without any accumulation
            if last_epoch == True and os.path.exists(save_path):
                cv2.imwrite(save_path, att)
                continue

            # naive filter out the low quality attention map with prob
            if prob < 0.1:  
                continue

            if not os.path.exists(save_path):
                cv2.imwrite(save_path, att)
            else:
                accu_att = cv2.imread(save_path, 0)
                accu_att = np.maximum(accu_att, att)
                cv2.imwrite(save_path,  accu_att)


def train_Integral_Attention_Model():
    pass


def hybrid_loss(prediction, label):
    """hybrid loss

    Args:
        prediction: models output (without activation -inf~inf).
        label: accumulated labels (0~1).

    Returns:
        returns hybrid loss

    """
    pos_loss = -tf.reduce_mean( tf.multiply('label>0', tf.math.log( tf.math.sigmoid('prediction (masked by) label>0') )) )

    neg_loss = -tf.reduce_mean( 
        -tf.math.log(
            tf.exp(
                -tf.nn.relu('prediction (masked by) label==0')
            )+1e-8
        ) 
        +
        tf.math.log(
            tf.math.exp(
                -tf.math.abs('prediction (masked by) label==0')
            )+1
        )
    ) 

    hybrid_loss = pos_loss + neg_loss
    return hybrid_loss

def make_accumulated_image_path(image_paths, image_dir, OAA_dir):
    """make_accumulated_path
    change image_dir to OAA_dir\n
    ex)\n
    image_paths = ["C://image//image.jpg", "C://image//image.jpg",...]\n
    image_dir = "C://image"\n
    OAA_dir = "C://OAA_sample"\n
    return = "C://OAA_sample//image.jpg", "C://OAA_sample//image.jpg",...]

    Args:
        image_path: source image paths
        image_dir: source image dir
        OAA_dir: save dir

    Returns:
        returns generated path

    """
    return [image_path.replace(image_dir, OAA_dir) for image_path in image_paths]

if __name__ == "__main__":
    pass