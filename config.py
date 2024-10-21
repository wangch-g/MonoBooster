import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='PoseDepth')

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# train mode params
mode_arg = add_argument_group('Train mode Params')
mode_arg.add_argument('--dataset',
                      type=str,
                      default="kitti_raw",
                      choices=["kitti_raw", "kitti_odom"],
                      help='')

# KITTI data params
kitti_arg = add_argument_group('KITTI data Params')
kitti_arg.add_argument('--kitti_raw_txt',
                       type=str,
                       default='./splits/eigen_zhou/train_files.txt',
                       help='Train set.')
kitti_arg.add_argument('--kitti_raw_root',
                       type=str,
                       default='',
                       help='/path/to/your/kitti/raw_data/root.')
kitti_arg.add_argument('--kitti_odom_txt',
                       type=str,
                       default='./splits/odom/train_files.txt',
                       help='Train set.')
kitti_arg.add_argument('--kitti_odom_root',
                       type=str,
                       default='',
                       help='/path/to/your/kitti/odometry/root.')
kitti_arg.add_argument('--kitti_hw',
                       type=tuple,
                       default=(192, 640),
                       choices=[(192, 640), (256, 832), (320, 1024)],
                       help='')
kitti_arg.add_argument('--img_ext',
                       type=str,
                       default='.jpg',
                       choices=['.png', '.jpg'],
                       help='')
kitti_arg.add_argument('--frame_ids',
                       nargs="+",
                       type=int,
                       default=[0, -1, 1],
                       help='')
kitti_arg.add_argument('--num_scales',
                       type=int,
                       default=4,
                       help='')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--batch_size',
                       type=int,
                       default=12,
                       help='# of images in each batch of data')
train_arg.add_argument('--num_workers',
                       type=int,
                       default=8,
                       help='# of subprocesses to use for data loading')
train_arg.add_argument('--pin_memory',
                       type=str2bool,
                       default=True,
                       help='# of subprocesses to use for data loading')
train_arg.add_argument('--shuffle',
                       type=str2bool,
                       default=True,
                       help='Whether to shuffle the train and valid indices')
train_arg.add_argument('--optim_policy',
                       type=str,
                       default='adam',
                       help='The optimation policy(adam or sgd).')
train_arg.add_argument('--start_epoch',
                       type=int,
                       default=0,
                       help='Number of epochs to train for.')
train_arg.add_argument('--max_epoch',
                       type=int,
                       default=20,
                       help='Number of epochs to train for.')
train_arg.add_argument('--init_lr',
                       type=float,
                       default=1e-4,
                       help='Initial learning rate value.')
train_arg.add_argument('--lr_factor',
                       type=float,
                       default=0.1,
                       help='Reduce learning rate value.')
train_arg.add_argument('--milestones',
                       type=list,
                       default=[15],
                       help='Reduce learning rate value.')
train_arg.add_argument('--display',
                       type=int,
                       default=50,
                       help='')

# data storage
storage_arg = add_argument_group('Storage')
storage_arg.add_argument('--train_log',
                         type=str,
                         default='train',
                         help='Training record.')
storage_arg.add_argument('--ckpt_dir',
                         type=str,
                         default='Res18_monobooster_640x192',
                         help='Training record.')

# depth net params
depth_net_arg = add_argument_group('DepthNet Params')
depth_net_arg.add_argument("--layer",
                           type=int,
                           default=18,
                           help='')

# depth loss params
depthloss_arg = add_argument_group('Depth net loss functions Params')
depthloss_arg.add_argument('--reproj_weight',
                           type=float,
                           default=1.0,
                           help='')
depthloss_arg.add_argument('--smooth_weight',
                           type=float,
                           default=0.001,
                           help='')
					   
# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--gpu',
                      type=int,
                      default=0,
                      help="Which GPU to run on.")										  
misc_arg.add_argument('--seed',
                      type=int,
                      default=1001,
                      help='Seed to ensure reproducibility.')					  
misc_arg.add_argument('--ckpt_root',
                      type=str,
                      default='./checkpoints',
                      help='Directory in which to save model checkpoints.')					  
misc_arg.add_argument('--logs_dir',
                      type=str,
                      default='./logs',
                      help='Directory in which logs wil be stored.')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
