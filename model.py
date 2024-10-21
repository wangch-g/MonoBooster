import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.resnet_encoder import ResnetEncoder
from networks.pose_decoder import PoseDecoder
from networks.depth_decoder import MonoBooster

from loss_functions import reprojection_loss_func
from loss_functions import disparity_smooth_loss_func

# from visualizer import Visualizer


class PoseDepth(nn.Module):
    def __init__(self, config):
        super(PoseDepth, self).__init__()
        self.config = config

        self.pose_encoder = ResnetEncoder(num_layers=18,
                                          pretrained=True,
                                          num_input_images=2)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=2)
        
        self.depth_encoder = ResnetEncoder(num_layers=config.layer,
                                           pretrained=True)
        self.depth_decoder = MonoBooster(self.depth_encoder.num_ch_enc,
                                         scales=range(self.config.num_scales),
                                         num_output_channels=1)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in range(self.config.num_scales):
            h = self.config.kitti_hw[0] // (2 ** scale)
            w = self.config.kitti_hw[1] // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.config.batch_size, h, w)
            self.backproject_depth[scale].to(torch.device("cuda:{}".format(self.config.gpu)))

            self.project_3d[scale] = Project3D(self.config.batch_size, h, w)
            self.project_3d[scale].to(torch.device("cuda:{}".format(self.config.gpu)))
    
    def pred_depth(self, img):
        feature = self.depth_encoder(img)
        outputs = self.depth_decoder(feature)
        
        disp = outputs['disp', 0]

        scaled_disp, depth = self.disp2depth(disp)

        return scaled_disp, depth
    
    def disp2depth(self, disp, min_depth=0.1, max_depth=100.0):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth
    
    def transformation_from_parameters(self, axisangle, translation, invert=False):
        """Convert the network's (axisangle, translation) output into a 4x4 matrix
        """
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()

        if invert:
            R = R.transpose(1, 2)
            t *= -1

        T = self.get_translation_matrix(t)

        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M

    def get_translation_matrix(self, translation_vector):
        """Convert a translation vector into a 4x4 transformation matrix
        """
        T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

        t = translation_vector.contiguous().view(-1, 3, 1)

        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t

        return T

    def rot_from_axisangle(self, vec):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be Bx1x3
        """
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)

        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca

        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)

        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1

        return rot
    
    def forward(self, inputs):  
        B, _, H, W = inputs[('color_aug', 0, 0)].shape
         
        data = {}
    
         # Predict depth (1./depth)
        data[('depth_feature', 0)] = self.depth_encoder(inputs[('color_aug', 0, 0)])
        data[('depth_outputs', 0)] = self.depth_decoder(data[('depth_feature', 0)])

        for id in self.config.frame_ids[1:]:
            # Predict pose
            if id < 0:
                pose_inputs = [inputs[('color_aug', id, 0)], inputs[('color_aug', 0, 0)]]
            else:
                pose_inputs = [inputs[('color_aug', 0, 0)], inputs[('color_aug', id, 0)]]
            
            pose_features = self.pose_encoder(torch.cat(pose_inputs, 1))
            axisangle, translation = self.pose_decoder([pose_features])
            pose44 = self.transformation_from_parameters(axisangle[:, 0],
                                                         translation[:, 0],
                                                         invert=(id < 0))
            data[('pose44_cur2ref', id)] = pose44
        
        losses = {}
        reproj_loss = 0.
        smooth_loss = 0.

        for s in range(self.config.num_scales):
            data[('disp', 0, s)] = F.interpolate(data[('depth_outputs', 0)]['disp', s],
                                                 size=(H, W),
                                                 mode='bilinear',
                                                 align_corners=False) # [B, 1, H, W]

            data[('scaled_disp', 0, s)], data[('depth', 0, s)] = self.disp2depth(data[('disp', 0, s)])
            
            reproj_loss_s = []
            iden_reproj_loss_s = []

            for id in self.config.frame_ids[1:]:
                cam_points = self.backproject_depth[0](data[('depth', 0, s)],
                                                       inputs[('inv_K', 0)])
                pix_coords = self.project_3d[0](cam_points,
                                                inputs[('K', 0)],
                                                data[('pose44_cur2ref', id)])
                data[("sample", id, s)] = pix_coords
                data[('warped_ref2cur', id, s)] = F.grid_sample(inputs[('color', id, 0)],
                                                                data[("sample", id, s)],
                                                                padding_mode="border",
                                                                align_corners=True)
                    
                # Compute loss
                reproj_loss_s.append(reprojection_loss_func(data[('warped_ref2cur', id, s)],
                                                            inputs[('color', 0, 0)]))
                iden_reproj_loss_s.append(reprojection_loss_func(inputs[('color', id, 0)],
                                                                 inputs[('color', 0, 0)])) 

            # Reprojection loss    
            reproj_loss_s = torch.cat(reproj_loss_s, dim=1)
            iden_reproj_loss_s = torch.cat(iden_reproj_loss_s, dim=1)
            # add random numbers to break ties
            iden_reproj_loss_s += torch.randn(iden_reproj_loss_s.shape).to(inputs[('color', 0, 0)].get_device()) * 0.00001

            combined = torch.cat((reproj_loss_s, iden_reproj_loss_s), dim=1)
            reproj_loss += torch.min(combined, dim=1)[0]

            # Smooth loss
            smooth_loss += disparity_smooth_loss_func(inputs[('color', 0, s)],
                                                      data[('depth_outputs', 0)]['disp', s] / (data[('depth_outputs', 0)]['disp', s].mean((2, 3), True) + 1e-12)) / (2 ** s)
            
        reproj_loss /= self.config.num_scales
        smooth_loss /= self.config.num_scales

        losses['reproj_loss'] = reproj_loss.mean()
        losses['smooth_loss'] = smooth_loss.mean()
        return losses
    

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords