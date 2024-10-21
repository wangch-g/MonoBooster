import torch
import torch.nn as nn

from einops.einops import rearrange

import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Pyramid depth-wise convolution
class PyramidDConv(nn.Module):
    def __init__(self, dim, dilation, bias):
        super(PyramidDConv, self).__init__()
        self.pad1 = nn.ReflectionPad2d(dilation[0])
        self.pad2 = nn.ReflectionPad2d(dilation[1])
        self.pad3 = nn.ReflectionPad2d(dilation[2])

        self.dwconv_d1 = nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=1, dilation=dilation[0], groups=dim//4, bias=bias)
        self.dwconv_d2 = nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=1, dilation=dilation[1], groups=dim//4, bias=bias)
        self.dwconv_d3 = nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=1, dilation=dilation[2], groups=dim//4, bias=bias)
        

    def forward(self, x):
        x, x1, x2, x3 = x.chunk(4, dim=1) 
        x1 = self.dwconv_d1(self.pad1(x1))
        x2 = self.dwconv_d2(self.pad2(x2))
        x3 = self.dwconv_d3(self.pad3(x3))
        x = torch.cat((x, x1, x2, x3), 1)

        return x
    

class CrossLevelAttention(nn.Module):
    def __init__(self, dim, dim_h, dim_l, num_heads, dilation=[1, 2, 3], bias=False, sque=4):
        super(CrossLevelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim//sque*2, kernel_size=1, bias=bias)
        self.kv_dwconv = PyramidDConv(dim//sque*2, dilation=dilation, bias=bias)

        self.q_h = nn.Conv2d(dim_h, dim_h//sque, kernel_size=1, bias=bias)
        self.q_h_dwconv = PyramidDConv(dim_h//sque, dilation=dilation, bias=bias)

        self.q_l = nn.Conv2d(dim_l, dim_l//sque, kernel_size=1, bias=bias)
        self.q_l_dwconv = PyramidDConv(dim_l//sque, dilation=dilation, bias=bias)

        self.proj_h = nn.Conv2d(dim_h//sque, dim_h, kernel_size=1, bias=bias)
        self.proj_l = nn.Conv2d(dim_l//sque, dim_l, kernel_size=1, bias=bias)

    def forward(self, x, x_h, x_l):
        _,_,h,w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q_h = self.q_h_dwconv(self.q_h(x_h))
        q_l = self.q_l_dwconv(self.q_l(x_l))
        
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_h = rearrange(q_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_l = rearrange(q_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k = torch.nn.functional.normalize(k, dim=-1)
        q_h = torch.nn.functional.normalize(q_h, dim=-1)
        q_l = torch.nn.functional.normalize(q_l, dim=-1)
        
        attn_h = (q_h @ k.transpose(-2, -1)) * self.temperature
        attn_h = attn_h.softmax(dim=-1)
        msg_h = (attn_h @ v)
        msg_h = rearrange(msg_h, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        msg_h = self.proj_h(msg_h)

        attn_l = (q_l @ k.transpose(-2, -1)) * self.temperature
        attn_l = attn_l.softmax(dim=-1)
        msg_l = (attn_l @ v)
        msg_l = rearrange(msg_l, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        msg_l = self.proj_l(msg_l)

        return msg_h, msg_l
