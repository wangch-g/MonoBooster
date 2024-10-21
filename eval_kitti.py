import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import eval_depth

from model import PoseDepth
import torch
import cv2
import numpy as np

from visualizer import Visualizer
visualizer = Visualizer(save_dir='./evaluation_results/eigen/vis')

def disp2depth(disp, min_depth=0.001, max_depth=80.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def resize_depths(gt_depth_list, pred_disp_list):
    gt_disp_list = []
    pred_depth_list = []
    pred_disp_resized = []
    for i in range(len(pred_disp_list)):
        h, w = gt_depth_list[i].shape
        pred_disp = cv2.resize(pred_disp_list[i], (w,h))
        pred_depth = 1.0 / (pred_disp + 1e-4)
        pred_depth_list.append(pred_depth)
        pred_disp_resized.append(pred_disp)
    
    return pred_depth_list, pred_disp_resized

def test_eigen_depth(cfg, model):
    print('Evaluate depth using eigen split.')
    filenames = open('./splits/eigen/test_files.txt').readlines()
    pred_disp_list = []
    for i in range(len(filenames)):
        path1, idx, _ = filenames[i].strip().split(' ')
        img = cv2.imread(os.path.join(os.path.join(cfg.raw_base_dir, path1), 'image_02/data/'+str(idx)+'.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, (cfg.kitti_hw[1], cfg.kitti_hw[0]))
        img_input = torch.from_numpy(img_resize / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        disp, _ = model.pred_depth(img_input)
        disp = disp[0].detach().cpu().numpy()
        disp = disp.transpose(1,2,0)
        # visualizer.save_disp_color_img(disp[:, :, 0], name='{}_disp'.format(i))
        # visualizer.save_img(img_input[0].mul(255).permute(1, 2, 0).byte().cpu().numpy(), '{}_img'.format(i))
        pred_disp_list.append(disp)
    
    gt_depths = np.load('./splits/eigen/gt_depths.npz', allow_pickle=True)['data']
    pred_depths, _ = resize_depths(gt_depths, pred_disp_list)
    eval_depth_res = eval_depth(gt_depths, pred_depths)
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_depth_res
    sys.stderr.write(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".
        format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                'a1', 'a2', 'a3'))
    sys.stderr.write(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".
        format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))

    return eval_depth_res

def resize_disp(pred_disp_list, gt_depths):
    pred_depths = []
    h, w = gt_depths[0].shape[0], gt_depths[0].shape[1]
    for i in range(len(pred_disp_list)):
        disp = pred_disp_list[i]
        resize_disp = cv2.resize(disp, (w,h))
        depth = 1.0 / resize_disp
        pred_depths.append(depth)
    
    return pred_depths

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description="Testing.")
    arg_parser.add_argument('--gpu', type=int, default=0, help='gpu id.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--layer', type=int, default=18, help='Resnet encoder type of depthnetwork.')
    arg_parser.add_argument('--raw_base_dir', type=str, default=None, help='')
    arg_parser.add_argument('--kitti_hw', type=int, default=[192, 640], nargs='+', help='')
    arg_parser.add_argument('--batch_size', type=int, default=1, help='')
    arg_parser.add_argument('--num_scales', type=int, default=4, help='')
    
    cfg = arg_parser.parse_args()

    model = PoseDepth(cfg)

    torch.cuda.set_device(cfg.gpu)
    model.cuda()
    weights = torch.load(cfg.pretrained_model, map_location='cuda:{}'.format(cfg.gpu))
    
    model.load_state_dict(weights['model_state'])
    model.eval()
    print('Model Loaded.')

    depth_res = test_eigen_depth(cfg, model)

