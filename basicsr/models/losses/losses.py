import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from .focal_frequency_loss import FocalFrequencyLoss as FFL
from torchvision import transforms

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# AT loss
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)
    
# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


# Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss




class CombinedLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(CombinedLoss, self).__init__()
        self.loss_weight = loss_weight
        self.psnr_loss = PSNRLoss()
        self.charbonnier_loss = CharbonnierLoss()
        self.ffl_loss = FFL(loss_weight=1.0, alpha=1.0)
        
        self.alpha = 1 
        self.beta = 50    
        self.gamma = 500
    def forward(self, pred, target):
        psnr = self.psnr_loss(pred, target)
        charbonnier = self.charbonnier_loss(pred, target)
        ffl = self.ffl_loss(pred, target)
        #print('****psnr Loss:', psnr)
        #print('charbonnier Loss:', charbonnier)
        #print('ffl Loss:', ffl)


        total_loss = self.alpha * psnr + self.beta * charbonnier + self.gamma * ffl
        return total_loss





def gaussian_kernel(window_size, sigma):
    """Generate a Gaussian kernel."""
    kernel = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
    kernel /= kernel.sum()
    return kernel.view(-1, 1)  # Ensures shape (N, 1)


def create_window(window_size, channel):
    """Create a 2D Gaussian window for image processing."""
    _1D_window = gaussian_kernel(window_size, 1.5)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, loss_weight=1.0, reduction='mean', toY=False):
        super(SSIMLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):

        _, channel, _, _ = pred.size()


        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred, target = pred / 255., target / 255.

        window = create_window(self.window_size, channel).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        loss = 1 - ssim_map.mean()
        return self.loss_weight * loss



class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class EDGELoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(EDGELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        # print(self.loss_weight)
        self.reduction = reduction
        # 定义Sobel算子
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_x = self.sobel_kernel_x.cuda()
        self.sobel_kernel_y = self.sobel_kernel_y.cuda()
        
    def edge(self, tensor_image):
        
        transform = transforms.Compose([           
            transforms.Grayscale(num_output_channels=1)  # 将图像转换为灰度图像
        ])
        
        tensor_image = transform(tensor_image)
        # print(tensor_image.shape)
        

        # 应用Sobel算子
        edge_x = F.conv2d(tensor_image, self.sobel_kernel_x, padding=1)
        edge_y = F.conv2d(tensor_image, self.sobel_kernel_y, padding=1)

        # 计算边缘强度
        edges = torch.sqrt(edge_x**2 + edge_y**2)

        # 将结果转换为0-1范围
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        return edges

    def forward(self, pred, gt, voxel, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred.shape)
        # print(gt.shape)
        # print(voxel.shape)
        # print(self.edge(pred).shape)
        # print(voxel[:,2,:,:]*voxel[:,3,:,:])
        # import matplotlib.pyplot as plt
        # plt.imshow(((voxel[1,2,:,:]*voxel[1,3,:,:])>0).cpu().numpy(), cmap='gray')
        # plt.savefig('/data2/yhe/NTIRE2025_EventDeblur_challenge-main/event.png')
        pred_edge = self.edge(pred)
        # plt.imshow(((pred_edge[1,0,:,:])>0).cpu().numpy(), cmap='gray')
        # plt.savefig('/data2/yhe/NTIRE2025_EventDeblur_challenge-main/edge.png')
        
        voxel_edge = (voxel[:,2,:,:]*voxel[:,3,:,:]).unsqueeze(1)
        # print(pred_edge.shape,voxel_edge.shape)
        
        loss=(((voxel_edge>0.01)*pred_edge-voxel_edge)**2).sum((1,2,3))#*self.loss_weight
        # print(type(self.loss_weight))
        # loss=loss*self.loss_weight
        
        if self.reduction == 'mean':
            loss = loss.mean()*float(self.loss_weight)
        elif self.reduction == 'sum':
            loss = loss.sum()*float(self.loss_weight)
        
        return loss,None
        
class SRNLoss(nn.Module):

    def __init__(self):
        super(SRNLoss, self).__init__()  

    def forward(self, preds, target):

        gt1 = target
        B,C,H,W = gt1.shape
        gt2 = F.interpolate(gt1, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, size=(H // 4, W // 4), mode='bilinear', align_corners=False)

        l1 = mse_loss(preds[0] , gt3)
        l2 = mse_loss(preds[1] , gt2)
        l3 = mse_loss(preds[2] , gt1)

        return l1+l2+l3



class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).
    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.
        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss
