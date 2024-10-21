import os
import cv2
import numpy as np
import matplotlib

from PIL import Image
from matplotlib import cm

class Visualizer():
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def save_img(self, img, name):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.save_dir, name+'.jpg'), img)

    def save_depth_img(self, depth, name):
        # depth([H, W, 1]])
        minddepth = np.min(depth)
        maxdepth = np.max(depth)
        depth_nor = (depth-minddepth) / (maxdepth-minddepth) * 255.0
        depth_nor = depth_nor.astype(np.uint8)
        cv2.imwrite(os.path.join(self.save_dir, name+'.jpg'), depth_nor)

    def save_disp_color_img(self, disp, name):
        # disp[H, W]
        vmax = np.percentile(disp, 95)
        normalizer = matplotlib.colors.Normalize(vmin=disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp)[:,:,:3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        
        name_dest_im = os.path.join(self.save_dir, name + '.jpg')
        im.save(name_dest_im)