import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from .spectralNormalization import SpectralNorm
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator

###############################################################################
# Helper Functions
###############################################################################

from torch.jit import Final
from timm.layers import use_fused_attn
from timm.models.layers import to_2tuple
import json


from rational_kat_cu.kat_rational import KAT_Group
from kan_convs import KANConv2DLayer


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, n_blocks=9, opt=opt)
    elif netG == 'kat':
        net = KATGenerator(input_nc, output_nc, ngf, opt=opt)
    elif netG == 'convkat':
        net = ConvKATGenerator(input_nc, output_nc, ngf, opt=opt)
    elif netG == 'kat_adv':
        net = KATGenerator_Advanced(input_nc, output_nc, ngf, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, opt=opt)
    elif netG == 'smallstylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=2, opt=opt)
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst', nl_layer='relu')
    elif netG == 'snsp':
        net = SNSPGAN_Generator(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        # net = NLayerDiscriminator(input_nc, ndf)
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif 'stylegan2' in netD:
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        elif gan_mode == "hinge":
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        elif self.gan_mode == 'hinge':
            if target_is_real:
                minvalue = torch.min(prediction - 1, torch.zeros(prediction.shape).to(prediction.device))
                loss = -torch.mean(minvalue)
            else:
                minvalue = torch.min(-prediction - 1,torch.zeros(prediction.shape).to(prediction.device))
                loss = -torch.mean(minvalue)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)

class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class G_Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon

##################################################################################
# Encoder and Decoders
##################################################################################


class E_adaIN(nn.Module):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        # style encoder
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def forward(self, image):
        style = self.enc_style(image)
        return style


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
        else:
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x, nce_layers=[], encode_only=False):
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None

        for layer_id, layer in enumerate(self.model):
            print(layer_id, layer)


class Decoder_all(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [Upsample2(scale_factor=2), Conv2dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample2(scale_factor=2), Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)

##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(True),
                        nn.ReflectionPad2d(1),
                        SpectralNorm(nn.Conv2d(in_features, in_features, 3)),
                        SCBottleneck(in_features, in_features)
                        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, opt=None):

        super(ResnetGenerator, self).__init__()
        self.opt = opt

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            SCBottleneck(ngf, ngf)
        )

        # Downsampleing layer 2 & 3
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            SCBottleneck(128, 128)
        )

        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            SCBottleneck(256, 256)
        )

        # Residual blocks      add 9 ResNet blocks
        res = []
        for _ in range(n_blocks):
            res += [ResidualBlock(256)]
        self.res = nn.Sequential(*res)


        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            SCBottleneck(128, 128)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            SCBottleneck(64, 64)
        )

        # Output layer
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, output_nc, 7),
            nn.Tanh()
         )


    def forward(self, input, layers=[], encode_only=False):
        layers1 = [0]
        layers2 = [0, 4]

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.conv_1):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers1:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_2):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers1:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_3):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers1:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.res):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers2:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    pass
            if encode_only:
                # print('encoder only return features')
                return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            # Encoding
            # batch size x 64 x 256 x 256
            c1 = self.conv_1(input)

            # batch size x 128 x 128 x 128
            c2 = self.conv_2(c1)

            # batch size x 256 x 64 x 64
            c3 = self.conv_3(c2)

            # Residual blocks * 9
            c4 = self.res(c3)

            # Decoding
            # batch size x 512 x 64 x 64
            skip1_de = torch.cat((c3, c4), 1)

            # batch size x 128 x 128 x 128
            c1_de = self.conv_5(skip1_de)

            # batch size x 256 x 128 x 128
            skip2_de = torch.cat((c2, c1_de), 1)

            # batch size x 64 x 256 x 256
            c3_de = self.conv_6(skip2_de)

            # batch size x 128 x 256 x 256
            skip3_de = torch.cat((c1, c3_de), 1)

            # batch size x 3 x 256 x 256
            fake = self.conv_7(skip3_de)
            return fake



# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None, flatten_=True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_ = flatten_

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

        # self.ident = nn.Identity()  

    def forward(self, x):

        # Patch embedding
        x = self.proj(x)

        # Flatten: BCHW -> BNC
        if self.flatten_:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)

        return x


# Patch UnEmbedding (convert to original shape)
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):

        # BNC -> BCHW
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)

        # Conv + PixelShuffle
        x = self.proj(x)
        return x


# Kolmogorov–Arnold Transformer: "https://github.com/Adamdad/kat/blob/main/katransformer.py"
class KAN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            # act_layer=KAT_Group,
            act_layer=None,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            act_init="gelu",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = functools.partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act2 = KAT_Group(mode=act_init)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(Attention, self).__init__()
        
        self.num_heads = num_heads
        self.input_dim = input_dim
        
        # Dimension per head
        self.head_dim = input_dim // num_heads
        
        assert self.head_dim * num_heads == input_dim, "Input dimension must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        
        # Output linear layer after attention
        self.out_linear = nn.Linear(input_dim, input_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: (B, N, C)  -> Batch size, Token length, Channels (input_dim)

        # Step 1: Linear projections for Q, K, V
        Q = self.query_linear(x)  # (B, N, C)
        K = self.key_linear(x)    # (B, N, C)
        V = self.value_linear(x)  # (B, N, C)

        # Step 2: Reshape Q, K, V to (B, num_heads, N, head_dim)
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)

        # Step 3: Scaled dot-product attention
        # Calculate attention scores: Q * K^T (transpose K)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, num_heads, N, N)

        # Scale the attention scores by the square root of head_dim
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # Step 4: Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)

        # Apply dropout (optional)
        attn_weights = self.dropout(attn_weights)

        # Step 5: Compute the output: Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)

        # Step 6: Concatenate the heads and pass through the output linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.input_dim)  # (B, N, C)
        output = self.out_linear(attn_output)  # (B, N, C)

        # Step 7: Add residual connection and layer normalization
        output = self.layer_norm(output + x)  # (B, N, C)
        
        return output



class KATBlock(nn.Module):
    def __init__(self, 
            patch_size=8, 
            in_chans=256, 
            out_chans=256, 
            embed_dim=256,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KAN,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.5,
            act_init: str = 'gelu',
            num_heads=8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.5,
            use_attn: bool = False,
            use_kan: bool = True,
        ):
        super(KATBlock, self).__init__()

        self.patchemd = PatchEmbed(patch_size, in_chans, embed_dim)

        self.norm1 = norm_layer(embed_dim)

        if use_attn:
            self.mixer = Attention(
                input_dim=embed_dim, 
                num_heads=num_heads,
            )
        else:
            self.mixer = nn.Identity()

        self.norm2 = norm_layer(embed_dim)

        if use_kan: # Replace MLP with KAN
            self.mlp = mlp_layer(
                in_features=embed_dim,
                hidden_features=int(embed_dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                act_init=act_init,
            )
        else: # Use a simple MLP
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.ReLU(True),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            )

        self.patchunemb = PatchUnEmbed(patch_size, out_chans, embed_dim)

    def forward(self, x):

        x = self.patchemd(x)
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.patchunemb(x)
        return x



class DoubleKATBlock(nn.Module):
    def __init__(self, 
            patch_size=8, 
            in_chans=256, 
            out_chans=256, 
            embed_dim=256,
            act_init: str = 'gelu',
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KAN,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.,
            # act_init: str = 'gelu',
        ):
        super(DoubleKATBlock, self).__init__()

        self.patchemd = PatchEmbed(patch_size, in_chans, embed_dim)

        self.norm1 = norm_layer(embed_dim)
        self.mixer = mlp_layer(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            act_init=act_init,
        )                          

        self.norm2 = norm_layer(embed_dim)
        self.mlp = mlp_layer(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            act_init=act_init,
        )

        self.patchunemb = PatchUnEmbed(patch_size, out_chans, embed_dim)

    def forward(self, x):

        x = self.patchemd(x)
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.patchunemb(x)
        return x



class TripleKANBlock(nn.Module):
    def __init__(self, 
            patch_size=8, 
            in_chans=256, 
            out_chans=256, 
            embed_dim=256,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KAN,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.,
            act_init: str = 'gelu',
        ):
        super(TripleKANBlock, self).__init__()

        self.patchemd = PatchEmbed(patch_size, in_chans, embed_dim)

        self.norm1 = norm_layer(embed_dim)
        self.mixer1 = mlp_layer(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            act_init=act_init,
        )                          

        # self.norm2 = norm_layer(embed_dim)
        # self.mixer2 = mlp_layer(
        #     in_features=embed_dim,
        #     hidden_features=int(embed_dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        #     act_init=act_init,
        # )

        # self.norm3 = norm_layer(embed_dim)
        # self.mixer3 = mlp_layer(
        #     in_features=embed_dim,
        #     hidden_features=int(embed_dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        #     act_init=act_init,
        # )

        # self.norm4 = norm_layer(embed_dim)
        # self.mixer4 = mlp_layer(
        #     in_features=embed_dim,
        #     hidden_features=int(embed_dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        #     act_init=act_init,
        # )

        self.patchunemb = PatchUnEmbed(patch_size, out_chans, embed_dim)

    def forward(self, x):

        x = self.patchemd(x)
        x = x + self.mixer1(self.norm1(x))
        # x = x + self.mixer2(self.norm2(x))
        # x = x + self.mixer3(self.norm3(x))
        # x = x + self.mixer4(self.norm4(x))
        x = self.patchunemb(x)
        return x


# For modifications:
class KATGenerator(nn.Module):

    def __init__(self, 
            input_nc, 
            output_nc, 
            ngf, 
            opt=None,
        ):

        super(KATGenerator, self).__init__()

        # self.opt = opt
        # self.patch_size = patch_size
        self.patch_size = opt.patch_size
        self.n_kat_blocks = opt.n_kat_blocks
        self.mixertype = opt.mixer
        self.act_type = opt.act_type
        # self.iokan = opt.iokan
        self.fks = opt.fks
        fpad = (self.fks-1)//2

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(input_nc, ngf, kernel_size=self.fks, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.SiLU(),
            SCBottleneck(ngf, ngf)
        )

        # Downsampleing layer 2 & 3
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.SiLU(),
            SCBottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.SiLU(),
            SCBottleneck(ngf*4, ngf*4)
        )


        # ======================================================

        # KAT blocks

        res = []
        for _ in range(self.n_kat_blocks):

            if self.mixertype == 'identity':
                res += [KATBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
            elif self.mixertype == 'kan':
                # res += [DoubleKATBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
                res += [DoubleKATBlock(self.patch_size, ngf*4, ngf*4, ngf*4, self.act_type)]
            elif self.mixertype == 'triplekan':
                res += [TripleKANBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
                # res += [TripleKANBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
            else:
                raise NotImplementedError('Generator is not recognized')

        self.res = nn.Sequential(*res)

        # self.res - nn.Identity()

        # 9 ResNet blocks
        # res = []
        # for _ in range(9):
        #     res += [ResidualBlock(ngf*4)]
        # self.res = nn.Sequential(*res)

        # ======================================================


        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.SiLU(),
            SCBottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.SiLU(),
            SCBottleneck(ngf, ngf)
        )

        # Output layer (original)
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(ngf*2, output_nc, self.fks),
            nn.Tanh()
        )


    def forward(self, input, layers=[], encode_only=False):

        layers1 = [0]
        layers2 = [0, 4]

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.conv_1):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_2):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_3):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.res):
                feat = layer(feat)
                if layer_id in layers2:
                    feats.append(feat)
                else:
                    pass
            if encode_only:
                return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            # Encoding
            # batch size x 64 x 256 x 256
            c1 = self.conv_1(input)

            # batch size x 128 x 128 x 128
            c2 = self.conv_2(c1)

            # batch size x 256 x 64 x 64
            c3 = self.conv_3(c2)

            # KAT blocks
            c4 = self.res(c3)

            # Decoding
            # batch size x 512 x 64 x 64
            skip1_de = torch.cat((c3, c4), 1)

            # batch size x 128 x 128 x 128
            c1_de = self.conv_5(skip1_de)

            # batch size x 256 x 128 x 128
            skip2_de = torch.cat((c2, c1_de), 1)

            # batch size x 64 x 256 x 256
            c3_de = self.conv_6(skip2_de)

            # batch size x 128 x 256 x 256
            skip3_de = torch.cat((c1, c3_de), 1)

            # batch size x 3 x 256 x 256
            fake = self.conv_7(skip3_de)
            return fake



class SCConv_SiLU(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv_SiLU, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.SiLU(),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck_SiLU(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck_SiLU, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.SiLU(),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv_SiLU(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.silu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.silu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.silu(out)

        return out



# Self-calibration with KA + SiLU
class SCKAConv_SiLU(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCKAConv_SiLU, self).__init__()

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Average pooling to (B, C, 1, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r)

        # self.patchemd = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patchemd = PatchEmbed(1, planes, planes)

        self.norm = nn.LayerNorm(planes)

        self.kanlayer = KAN(
                in_features=planes,
                hidden_features=int(planes * 4),
                act_layer=nn.GELU,
                drop=0.,
                act_init='gelu',
        )

        self.patchunemb = PatchUnEmbed(1, planes, planes)

        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.SiLU(),
        )

    def forward(self, x):

        identity = x                        # B C H W

        b, c, h, w = identity.size()

        x = self.avg_pool(x)                # B C H' W'

        x = self.patchemd(x)                    # B C H' W' -> B N C
        x = x + self.kanlayer(self.norm(x))     # B N C
        x = self.patchunemb(x)                  # B N C -> B C H' W'

        x = F.interpolate(x, identity.size()[2:])       # B C H' W' -> B C H W

        x = torch.add(identity, x)

        x = torch.sigmoid(x)

        x = torch.mul(self.k3(identity), x)

        x = self.k4(x)  # k4

        return x



class SCKABottleneck_SiLU(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCKABottleneck_SiLU, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.SiLU(),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCKAConv_SiLU(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.silu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.silu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.silu(out)

        return out



class KATGenerator_Advanced(nn.Module):

    def __init__(self, 
            input_nc, 
            output_nc, 
            ngf, 
            opt=None,
        ):

        super(KATGenerator_Advanced, self).__init__()

        # self.opt = opt
        # self.patch_size = patch_size
        self.patch_size = opt.patch_size
        self.n_kat_blocks = opt.n_kat_blocks
        self.mixertype = opt.mixer
        self.act_type = opt.act_type
        # self.iokan = opt.iokan
        self.fks = opt.fks
        fpad = (self.fks-1)//2

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(input_nc, ngf, kernel_size=self.fks, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.SiLU(),
            SCBottleneck(ngf, ngf)
            # SCBottleneck_SiLU(ngf, ngf)
        )

        # Downsampleing layer 2 & 3
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.SiLU(),
            SCBottleneck(ngf*2, ngf*2)
            # SCBottleneck_SiLU(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.SiLU(),
            SCBottleneck(ngf*4, ngf*4)
            # SCKABottleneck_SiLU(ngf*4, ngf*4)           # dehaze36
            # SCBottleneck_SiLU(ngf*4, ngf*4)           # dehaze35
        )


        # ======================================================
        # KAT blocks

        res = []
        for _ in range(self.n_kat_blocks):

            if self.mixertype == 'identity':
                res += [KATBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
            elif self.mixertype == 'kan':
                res += [DoubleKATBlock(self.patch_size, ngf*4, ngf*4, ngf*4, self.act_type)]
            elif self.mixertype == 'triplekan':
                res += [TripleKANBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
            else:
                raise NotImplementedError('Generator is not recognized')

        self.res = nn.Sequential(*res)

        # ======================================================


        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.SiLU(),
            SCBottleneck(ngf*2, ngf*2)
            # SCBottleneck_SiLU(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.SiLU(),
            SCBottleneck(ngf, ngf)
            # SCBottleneck_SiLU(ngf, ngf)
        )

        # Output layer (original)
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(ngf*2, output_nc, self.fks),
            nn.Tanh()
        )


    def forward(self, input, layers=[], encode_only=False):

        layers1 = [0]
        layers2 = [0, 4]

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.conv_1):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_2):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_3):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.res):
                feat = layer(feat)
                if layer_id in layers2:
                    feats.append(feat)
                else:
                    pass
            if encode_only:
                return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            # Encoding
            # batch size x 64 x 256 x 256
            c1 = self.conv_1(input)

            # batch size x 128 x 128 x 128
            c2 = self.conv_2(c1)

            # batch size x 256 x 64 x 64
            c3 = self.conv_3(c2)

            # KAT blocks
            c4 = self.res(c3)

            # Decoding
            # batch size x 512 x 64 x 64
            skip1_de = torch.cat((c3, c4), 1)

            # batch size x 128 x 128 x 128
            c1_de = self.conv_5(skip1_de)

            # batch size x 256 x 128 x 128
            skip2_de = torch.cat((c2, c1_de), 1)

            # batch size x 64 x 256 x 256
            c3_de = self.conv_6(skip2_de)

            # batch size x 128 x 256 x 256
            skip3_de = torch.cat((c1, c3_de), 1)

            # batch size x 3 x 256 x 256
            fake = self.conv_7(skip3_de)
            return fake



class SCKAConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCKAConv, self).__init__()

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Average pooling to (B, C, 1, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r)

        # self.patchemd = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patchemd = PatchEmbed(1, planes, planes)

        self.norm = nn.LayerNorm(planes)

        self.kanlayer = KAN(
                in_features=planes,
                hidden_features=int(planes * 4),
                act_layer=nn.GELU,
                drop=0.,
                act_init='gelu',
        )

        self.patchunemb = PatchUnEmbed(1, planes, planes)

        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):

        identity = x                        # B C H W

        b, c, h, w = identity.size()

        x = self.avg_pool(x)                # B C H' W'

        x = self.patchemd(x)                    # B C H' W' -> B N C
        x = x + self.kanlayer(self.norm(x))     # B N C
        x = self.patchunemb(x)                  # B N C -> B C H' W'

        x = F.interpolate(x, identity.size()[2:])       # B C H' W' -> B C H W

        x = torch.add(identity, x)

        x = torch.sigmoid(x)

        x = torch.mul(self.k3(identity), x)

        x = self.k4(x)  # k4

        return x



class SCKABottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_channels, out_channels):
        super(SCKABottleneck, self).__init__()

        planes = int(in_channels / 2)
        # planes = int(in_channels / 4)
        # pooling_r = 4

        self.conv1_a = nn.Conv2d(in_channels, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_channels, planes, 1, 1)

        self.scconv = SCKAConv(planes, self.pooling_r)
        # self.scconv = SCKAConv(planes, pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, out_channels, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out



class ConvKATGenerator(nn.Module):

    def __init__(self, 
            input_nc, 
            output_nc, 
            ngf, 
            opt=None,
        ):

        super(ConvKATGenerator, self).__init__()

        self.patch_size = opt.patch_size
        self.n_kat_blocks = opt.n_kat_blocks
        self.mixertype = opt.mixer
        self.act_type = opt.act_type
        self.fks = opt.fks
        fpad = (self.fks-1)//2

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(input_nc, ngf, kernel_size=self.fks, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            SCBottleneck(ngf, ngf)
            # SCKABottleneck(ngf, ngf)
        )

        # Downsampleing layer 2 & 3
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            SCBottleneck(ngf*2, ngf*2)
            # SCKABottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
            SCKABottleneck(ngf*4, ngf*4)
        )


        # ======================================================

        # KAT blocks

        res = []
        for _ in range(self.n_kat_blocks):

            if self.mixertype == 'identity':
                res += [KATBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
            elif self.mixertype == 'kan':
                # res += [DoubleKATBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
                res += [DoubleKATBlock(self.patch_size, ngf*4, ngf*4, ngf*4, self.act_type)]
            elif self.mixertype == 'triplekan':
                res += [TripleKANBlock(self.patch_size, ngf*4, ngf*4, ngf*4)]
            else:
                raise NotImplementedError('Generator is not recognized')

        self.res = nn.Sequential(*res)

        # self.res - nn.Identity()

        # ======================================================


        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            SCBottleneck(ngf*2, ngf*2)
            # SCKABottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            SCBottleneck(ngf, ngf)
            # SCKABottleneck(ngf, ngf)
        )

        # Output layer
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(ngf*2, output_nc, self.fks),
            nn.Tanh()
        )


    def forward(self, input, layers=[], encode_only=False):

        layers1 = [0]
        layers2 = [0, 4]

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.conv_1):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_2):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_3):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.res):
                feat = layer(feat)
                if layer_id in layers2:
                    feats.append(feat)
                else:
                    pass
            if encode_only:
                return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            # Encoding
            # batch size x 64 x 256 x 256
            c1 = self.conv_1(input)

            # batch size x 128 x 128 x 128
            c2 = self.conv_2(c1)

            # batch size x 256 x 64 x 64
            c3 = self.conv_3(c2)

            # KAT blocks
            c4 = self.res(c3)

            # Decoding
            # batch size x 512 x 64 x 64
            skip1_de = torch.cat((c3, c4), 1)

            # batch size x 128 x 128 x 128
            c1_de = self.conv_5(skip1_de)

            # batch size x 256 x 128 x 128
            skip2_de = torch.cat((c2, c1_de), 1)

            # batch size x 64 x 256 x 256
            c3_de = self.conv_6(skip2_de)

            # batch size x 128 x 256 x 256
            skip3_de = torch.cat((c1, c3_de), 1)

            # batch size x 3 x 256 x 256
            fake = self.conv_7(skip3_de)
            return fake



# ===============================================================================================================================

# ===============================================================================================================================


class ResnetDecoder(nn.Module):
    """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based decoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if(no_antialias):
                model += [SpectralNorm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2)),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1,
                                       bias=use_bias),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          SpectralNorm(nn.Conv2d(ngf * mult, int(ngf * mult / 2)),
                                       kernel_size=3, stride=1,
                                       padding=1,
                                       bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetEncoder(nn.Module):
    """Resnet-based encoder that consists of a few downsampling + several Resnet blocks
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)),
                          nn.ReLU(True)]
            else:
                model += [SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()        # Norm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(input)


class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)



# ======================= Cycle-SNSPGAN ======================= #

import math

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel, fidx_u=[0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3], fidx_v=[0, 1, 0, 5,
                                                                                2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 2, 5]):
    scale_ratio = width//7
    fidx_u = [u*scale_ratio for u in fidx_u]
    fidx_v = [v*scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * c_part: (i+1)*c_part, t_x, t_y]\
                =get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights


class FcaLayer(nn.Module):
    def __init__(self, channel, reduction, width, height):
        super(FcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights',get_dct_weights(self.width, self.height, channel))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x,(self.height,self.width))
        y = torch.sum(y*self.pre_computed_dct_weights,dim=(2,3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SNSPGAN_Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(SNSPGAN_Generator, self).__init__()

        # Encoding layers
        # Initial convolution block
        # inputsize:3*256*256, outputsize:64*256*256
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            FcaLayer(64, 8, 32, 32)
        )

        # Downsampling
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            FcaLayer(128, 8, 32, 32)
        )
        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            FcaLayer(256, 8, 32, 32)
        )

        # Residual blocks
        # inputsize:256*64*64, outputsize:256*64*64
        in_features = 256
        self.conv_41 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_features, in_features, 3),
                                     nn.InstanceNorm2d(in_features),
                                     nn.ReLU(inplace=True))

        self.conv_42 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_features, in_features, 3),
                                     nn.InstanceNorm2d(in_features),
                                     FcaLayer(in_features, 8, 32, 32))

        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            FcaLayer(128, 8, 32, 32)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            FcaLayer(64, 8, 32, 32)
        )

        # Output layer
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, output_nc, 7),
            nn.Tanh()
        )


    def forward(self, x):
        """ The forward pass of the generator with skip connections
                """
        # Encoding
        # batch size x 64 x 256 x 256
        c1 = self.conv_1(x)
        # c1_1 = self.cbam1(c1)

        # batch size x 128 x 128 x 128
        c2 = self.conv_2(c1)

        # batch size x 256 x 64 x 64
        c3 = self.conv_3(c2)

        # Residual blocks * 9
        # batch size x 256 x 64 x 64
        # c4_1 = self.conv_41(c3)
        # c4_2 = c3 + self.conv_42(c4_1)
        # c4_3 = self.conv_41(c4_2)
        # c4_4 = c4_2 + self.conv_42(c4_3)
        # c4_5 = self.conv_41(c4_4)
        # c4_6 = c4_4 + self.conv_42(c4_5)
        # c4_7 = self.conv_41(c4_6)
        # c4_8 = c4_6 + self.conv_42(c4_7)
        # c4_9 = self.conv_41(c4_8)
        # c4_10 = c4_8 + self.conv_42(c4_9)
        # c4_11 = self.conv_41(c4_10)
        # c4_12 = c4_10 + self.conv_42(c4_11)
        # c4_13 = self.conv_41(c4_12)
        # c4_14 = c4_12 + self.conv_42(c4_13)
        # c4_15 = self.conv_41(c4_14)
        # c4_16 = c4_14 + self.conv_42(c4_15)
        # c4_17 = self.conv_41(c4_16)
        # c4 = c4_16 + self.conv_42(c4_17)

        # double res
        c4_1 = c3 + self.conv_41(c3)
        c4_2 = c3 + self.conv_42(c4_1)
        c4_3 = c4_2 + self.conv_41(c4_2)
        c4_4 = c4_2 + self.conv_42(c4_3)
        c4_5 = c4_4 + self.conv_41(c4_4)
        c4_6 = c4_4 + self.conv_42(c4_5)
        c4_7 = c4_6 + self.conv_41(c4_6)
        c4_8 = c4_6 + self.conv_42(c4_7)
        c4_9 = c4_8 + self.conv_41(c4_8)
        c4_10 = c4_8 + self.conv_42(c4_9)
        c4_11 = c4_10 + self.conv_41(c4_10)
        c4_12 = c4_10 + self.conv_42(c4_11)
        c4_13 = c4_12 + self.conv_41(c4_12)
        c4_14 = c4_12 + self.conv_42(c4_13)
        c4_15 = c4_14 + self.conv_41(c4_14)
        c4_16 = c4_14 + self.conv_42(c4_15)
        c4_17 = c4_16 + self.conv_41(c4_16)
        c4 = c4_16 + self.conv_42(c4_17)

        # Decoding
        # batch size x 512 x 64 x 64
        skip1_de = torch.cat((c3, c4), 1)

        # batch size x 128 x 128 x 128
        c1_de = self.conv_5(skip1_de)

        # batch size x 256 x 128 x 128
        skip2_de = torch.cat((c2, c1_de), 1)

        # batch size x 64 x 256 x 256
        c3_de = self.conv_6(skip2_de)

        # batch size x 128 x 256 x 256
        skip3_de = torch.cat((c1, c3_de), 1)

        # batch size x 3 x 256 x 256
        c4_de = self.conv_7(skip3_de)

        return c4_de



class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )





class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)



class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y