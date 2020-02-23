import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ArgumentParser:
    def __init__(self,):
        pass
        
    def parse_args(self,):
        parser = argparse.ArgumentParser(description='FireClassifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        # preprocessing
        parser.add_argument('--multi_scale', dest='multi_scale', default=False, type=str2bool)
        parser.add_argument('--min_image_size', dest='min_image_size', help='min_image_size', default=64, type=int)
        parser.add_argument('--max_image_size', dest='max_image_size', help='max_image_size', default=224, type=int)
        
        parser.add_argument('--num_threads', dest='num_threads', default=6, type=int)
        parser.add_argument('--random_crop', dest='random_crop', help='random_crop', default=True, type=str2bool)
        
        # update !!
        parser.add_argument('--experimenter', dest='experimenter', help='experimenter', default='JSH', type=str)
        
        # for linux
        parser.add_argument('--root_dir', dest='root_dir', help='root_dir', default='../DB/Recon_FireClassifier_DB_20200219/', type=str)
        
        # for windows
        ## NAS
        # parser.add_argument('--root_dir', dest='root_dir', help='root_dir', default='//gynetworks/Data/DATA/Image/Recon_FireClassifier_DB_20200219/', type=str)
        
        ## Local
        # parser.add_argument('--root_dir', dest='root_dir', help='root_dir', default='C:/DB/Recon_FireClassifier_DB_20200219/', type=str)
        
        # gpu option
        parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
        parser.add_argument('--batch_size_per_gpu', dest='batch_size_per_gpu', default=32, type=int)
        
        # model option
        parser.add_argument('--option', dest='option', default='b0', type=str)
        
        # train technology
        parser.add_argument('--augment', dest='augment', help='None/weakaugment/randaugment', default=None, type=str)
        
        parser.add_argument('--mixup', dest='mixup', help='mixup', default=False, type=str2bool)
        parser.add_argument('--mixup_alpha', dest='mixup_alpha', help='mixup_alpha', default=1.0, type=float)

        parser.add_argument('--cutmix', dest='cutmix', help='cutmix', default=False, type=str2bool)
        
        parser.add_argument('--weight_decay', dest='weight_decay', help='weight_decay', default=1e-4, type=float)
        parser.add_argument('--ema_decay', dest='ema_decay', help='ema', default=-1, type=float)

        parser.add_argument('--log_iteration', dest='log_iteration', help='log_iteration', default=100, type=int)
        parser.add_argument('--val_iteration', dest='val_iteration', help='val_iteration', default=20000, type=int)
        parser.add_argument('--max_iteration', dest='max_iteration', help='max_iteration', default=200000, type=int)
        
        return vars(parser.parse_args())

    
