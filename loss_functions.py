import torch
import torch.nn as nn


class SSIM(nn.Module):
    """
    Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def reprojection_loss_func(warped, cur):
    """
    Computes reprojection loss between a batch of warped and cur images
    """
    ssim = SSIM()
    ssim.to(warped.get_device())
    abs_diff = torch.abs(cur - warped)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(warped, cur).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def reprojection_loss_func_sigma(warped, cur, sigma):
    """
    Computes reprojection loss between a batch of warped and cur images
    """
    ssim = SSIM()
    ssim.to(warped.get_device())
    abs_diff = torch.abs(cur - warped)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(warped, cur).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    # Reference: https://github.com/no-Seaweed/Learning-Deep-Learning-1/blob/master/paper_notes/sfm_learner.md
    transformed_sigma = (10 * sigma + 0.1)
    reprojection_loss = (reprojection_loss / transformed_sigma) + torch.log(transformed_sigma)

    return reprojection_loss

def disparity_smooth_loss_func(img, disp):
    """
    Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean((1, 2, 3)) + grad_disp_y.mean((1, 2, 3))