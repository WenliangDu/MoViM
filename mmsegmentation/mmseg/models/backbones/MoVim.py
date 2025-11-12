import math
import torch
from torch import nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from fvcore.nn import flop_count

from .MoVim_unit import Conv2d_BN, LayerNorm1D, LayerNorm2D, Stem, PatchMerging,ConvLayer1D,ConvLayer2D,InvertedResidual,ConvolutionalGLU,PyramidPoolAgg

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class GLF(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ratio:int,
        norm_cfg=dict(type='BN', requires_grad=True),
    ) -> None:
        super(GLF, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup*ratio, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.global_act = ConvModule(inp, oup*ratio, kernel_size=1, norm_cfg=None, act_cfg=None) 
        self.g = ConvModule(oup*ratio,oup,1,norm_cfg=norm_cfg,act_cfg=None)
        self.dwconv = ConvModule(oup,oup,3,1,1,groups=oup,norm_cfg=None)
        
        self.act = nn.Sigmoid()  # 直接使用 PyTorch 版本
    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        
        
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = self.dwconv(self.g(local_feat*global_act))+global_feat

        return out

class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0) 
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L= x.shape
        H = int(math.sqrt(L))
        
        BCdt = self.dw(self.BCdt_proj(x).view(batch,-1, H, H)).flatten(2)
        B,C,dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1) 
        A = (dt + self.A.view(1,-1,1)).softmax(-1) 
        
        AB = (A * B) 
        h = x @ AB.transpose(-2,-1) 
        
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) 
        h = self.out_proj(h * self.act(z)+ h * self.D)
        y = h @ C # B C N, B C L -> B C L
        
        y = y.view(batch,-1,H,H).contiguous()# + x * self.D  # B C H W
        return y, h

class Vim_CNN_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64,global_ratio=0.5,kernels=3,expand_ratio=3):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.global_channels = int(global_ratio * dim)
        self.local_channels = dim - self.global_channels
        
        self.mixer = HSMSSD(d_model=self.global_channels, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(self.global_channels)
       
        self.local_op = InvertedResidual(self.local_channels, self.local_channels, ks=kernels, stride=1, expand_ratio=expand_ratio)
        
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
       
        
     
        self.mlp = ConvolutionalGLU(in_features=dim,hidden_features=int(dim*mlp_ratio))
        
        
        #LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(2,dim), requires_grad=True)
        self.beta = nn.Parameter(1e-4 * torch.ones(1,self.global_channels), requires_grad=True)
        self.delta = nn.Parameter(1e-4 * torch.ones(1,self.local_channels), requires_grad=True)
        
        
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(2,-1,1,1)
        beta = torch.sigmoid(self.beta).view(1,-1,1,1)
        delta = torch.sigmoid(self.delta).view(1,-1,1,1)
        
        # DWconv1
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)
        
        global_x, local_x = torch.split(x, [self.global_channels, self.local_channels], dim=1)
        # HSM-SSD
        
        global_x_prev = global_x
        global_x, h = self.mixer(self.norm(global_x.flatten(2)))
        global_x = (1-beta) * global_x_prev + beta * global_x
        
        local_x_prev = local_x
        local_x = self.local_op(local_x)
        local_x = (1-delta) * local_x_prev + delta * local_x        
        
        x = torch.cat([global_x, local_x], dim=1)
        
        x = (1-alpha[1]) * x + alpha[1] * self.mlp(x)
        return x, h

class Vim_CNN_Stage(nn.Module):
    def __init__(self, in_dim, out_dim, depth,  mlp_ratio=4.,downsample=None, 
                 ssd_expand=1, state_dim=64,global_ratio=0.5,kernels=3,expand_ratio =3):
        super().__init__()
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            Vim_CNN_Block(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim,
                                  global_ratio=global_ratio,kernels=kernels,expand_ratio=expand_ratio) for _ in range(depth)])
        
        self.downsample = downsample(in_dim=in_dim, out_dim =out_dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x, h = blk(x)
            
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_out, h

class Vim_CNN_Backbone(nn.Module):
    def __init__(self, in_dim=3,Stem_dim=64,embed_dim=[64,128,160], depths=[2, 2, 2], mlp_ratio=4., ssd_expand=1,
                 state_dim=[49,25,9],global_ratio=[0.8,0.7,0.6],kernels=[5,3,5],expand_ratio=[3,3,6]):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_embed = Stem(in_dim=in_dim, dim=Stem_dim)
        PatchMergingBlock = PatchMerging

        # 构建层
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = Vim_CNN_Stage(
                in_dim=int(embed_dim[i_layer]),
                out_dim=int(embed_dim[i_layer+1]) if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                ssd_expand=ssd_expand,
                state_dim=state_dim[i_layer],
                global_ratio=global_ratio[i_layer],
                kernels=kernels[i_layer],
                expand_ratio=expand_ratio[i_layer]     
            )
            self.stages.append(stage)
        
        self.norms = nn.ModuleList([
            LayerNorm2D(embed_dim[0]),
            LayerNorm2D(embed_dim[1]),
            LayerNorm2D(embed_dim[2]),
        ])

    def forward(self, x):
        x1,x = self.patch_embed(x)

        outs = []
        outs.append(x1)
        for i, stage in enumerate(self.stages):
            x, x_out, h = stage(x)
            x_out = self.norms[i](x_out) 
            outs.append(x_out)  # 逐层存储特征


        return tuple(outs)  # 返回特征金字塔

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1) # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

class Block(nn.Module):

    def __init__(self, dim,key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)
        self.alpha = nn.Parameter(1e-4 * torch.ones(2,dim),requires_grad=True)
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))       
        alpha = torch.sigmoid(self.alpha).view(2,-1,1,1)
       
        x1 = (1 - alpha[0]) * x1 + alpha[0] * self.drop_path(self.attn(x1))
        x1 = (1 - alpha[1]) * x1 + alpha[1] * self.drop_path(self.mlp(x1))
        
        return x1
    
class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                norm_cfg=dict(type='BN2d', requires_grad=True), 
                act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x
    

@MODELS.register_module()
class MoVim(BaseModule):
    def __init__(self,
                 channels,
                 out_channels,
                 decode_out_indices=[1, 2, 3], 
                 in_channels=5,
                 depths=4,
                 key_dim=16,
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=2,
                 c2t_stride=2,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 injection_type="muli_sum",
                 init_cfg=None,
                 injection=True):
        super().__init__()
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices
        self.init_cfg = init_cfg
        self.in_channels = in_channels
        if self.init_cfg != None:
            self.pretrained = self.init_cfg['checkpoint']

        self.tpm = Vim_CNN_Backbone(
            in_dim=in_channels,Stem_dim=64,embed_dim=[64,128,160], depths=[2, 2, 2],global_ratio=[0.8,0.7,0.6],kernels=[7,5,3],expand_ratio=[3,3,5]
        )

        self.ppa = PyramidPoolAgg(stride=c2t_stride)
   
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0, attn_drop=0, 
            drop_path=dpr,
            norm_cfg=norm_cfg,
            act_layer=act_layer)
        self.SIM = nn.ModuleList()
        if self.injection:
            for i in range(len(channels)):
                if i in decode_out_indices:
                    self.SIM.append(
                        GLF(channels[i], out_channels[i],ratio=2,norm_cfg=norm_cfg))
                else:
                    self.SIM.append(nn.Identity())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    
    def forward(self, x):
        ouputs = self.tpm(x)
        
        out = self.trans(self.ppa(ouputs))

        if self.injection:

            xx = out.split(self.channels, dim=1)
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    local_tokens = ouputs[i]
                    global_semantics = xx[i]
                    out_ = self.SIM[i](local_tokens, global_semantics)
                    results.append(out_)
            return results
        else:
            ouputs.append(out)
            return ouputs
        
