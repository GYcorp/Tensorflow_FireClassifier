import tensorflow as tf

def flags_to_dict(flags):
    return {k : flags[k].value for k in flags}

def get_config():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    ###############################################################################
    # Default Config
    ###############################################################################
    # for windows
    # flags.DEFINE_string('root_dir', 'C:/DB/Recon_FireClassifier_DB_20200219/', 'unknown')
    
    # for linux
    flags.DEFINE_string('root_dir', '../DB/Recon_FireClassifier_DB_20200219/', 'unknown')

    flags.DEFINE_string('experimenter', 'JSH', 'unknown')
    flags.DEFINE_string('use_gpu', '0', 'unknown')
    
    ###############################################################################
    # Training Schedule
    ###############################################################################
    flags.DEFINE_float('init_learning_rate', 0.016, 'unknown')
    flags.DEFINE_float('alpha_learning_rate', 0.002, 'unknown')

    flags.DEFINE_integer('batch_size', 32, 'unknown')
    flags.DEFINE_integer('batch_size_per_gpu', 32, 'unknown')

    flags.DEFINE_integer('log_iteration', 100, 'unknown')
    flags.DEFINE_integer('valid_iteration', 20000, 'unknown')
    flags.DEFINE_integer('warmup_iteration', 10000, 'unknown')
    flags.DEFINE_integer('max_iteration', 200000, 'unknown')
    
    ###############################################################################
    # Training Technology
    ###############################################################################
    flags.DEFINE_boolean('mixup', False, 'unknown')
    flags.DEFINE_boolean('cutmix', False, 'unknown')
    flags.DEFINE_boolean('random_crop', False, 'unknown')

    flags.DEFINE_string('option', 'b0', 'unknown')
    flags.DEFINE_integer('image_size', 224, 'unknown')

    flags.DEFINE_string('augment', 'None', 'None/weakly_augment/randaugment')

    flags.DEFINE_float('weight_decay', 1e-4, 'unknown')

    # flags.DEFINE_boolean('ema', False, 'unknown')
    # flags.DEFINE_float('ema_decay', 0.999, 'unknown')

    return FLAGS

if __name__ == '__main__':
    import json
    
    flags = get_config()

    print(flags.use_gpu)
    print(flags_to_dict(flags))
    
    # print(flags.mixup)
    # print(flags.efficientnet_option)