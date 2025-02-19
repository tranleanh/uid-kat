import numpy as np
import torch
from .base_model import BaseModel, VGGNet
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

import torch.nn.functional as F

# pretrained VGG16 module set in evaluation mode for feature extraction
vgg = VGGNet().cuda().eval()


class UCLModel(BaseModel):
    """ This class implements UCL-Dehaze model

    The code borrows heavily from the PyTorch implementation of CycleGAN, CUT and CWR
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    https://github.com/taesungp/contrastive-unpaired-translation
    https://github.com/JunlinHan/CWR
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for lambda_NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=5.0, help='weight for NCE loss: IDT(G(Y), Y)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,5,9,13,17', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not UCL-Dehaze")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'idt', 'perceptual']              # Original
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'idt']
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'idt', 'perceptual', 'PHC'] 
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            # self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()      # LS_GAN MSE loss
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)    # input & generated
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = 0
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y)
            self.loss_idt = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_IDT       # G(Y) & clear w.CUT/ w.o. FastCUT
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_perceptual = self.perceptual_loss(self.real_A, self.fake_B, self.real_B) * 0.0002   


        # Original
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt + self.loss_perceptual     


        # ------- Added PHC loss -------
        # self.loss_PHA = self.calculate_PHA_loss(self.fake_B, self.real_B) * 0.0001
        # self.loss_PHC = self.calculate_PHC_loss(self.real_A, self.fake_B, self.real_B)


        # self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt + self.loss_perceptual + self.loss_PHC      # add PHC_loss
        # self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt + self.loss_PHC                             # add PHC_loss | remove perceptual_loss
        # self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt
        # print(self.loss_G_GAN, loss_NCE_both, self.loss_idt, self.loss_perceptual, self.loss_DCH)
        # ------------------------------

        return self.loss_G


    def perceptual_loss(self, x, y, z):
        # c = torch.nn.MSELoss()
        c = torch.nn.L1Loss()

        fx1, fx2, fx3 = vgg(x)      # hazy extract
        fy1, fy2, fy3 = vgg(y)      # dehazed
        fz1, fz2, fz3 = vgg(z)      # clear

        m1 = c(fz1, fy1) / c(fx1, fy1)
        m2 = c(fz2, fy2) / c(fx2, fy2)
        m3 = c(fz3, fy3) / c(fx3, fy3)

        loss = 0.4 * m1 + 0.6 * m2 + m3
        return loss


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)     # CUT: RGB,1,2(downsampling),1,5(resblock)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:      # Fast CUT (Cycle-GAN default)
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)               # feature
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)  # MLP 256 patch
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)     # MLP

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers



    # =========================================================================================================

    # def extract_patch(self, image, center, patch_size):
    #     # Get patch coordinates, ensuring they are within the bounds of the image
    #     C, H, W = image.shape[1], image.shape[2], image.shape[3]
    #     center_h, center_w = int(center[0]), int(center[1])
        
    #     # Calculate the crop box
    #     top = max(center_h - patch_size // 2, 0)
    #     left = max(center_w - patch_size // 2, 0)
    #     bottom = min(top + patch_size, H)
    #     right = min(left + patch_size, W)
        
    #     # Ensure the patch is the exact size required
    #     patch = image[:, top:bottom, left:right]
        
    #     # If the patch is smaller than the required size, we can pad it
    #     if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
    #         patch = F.pad(patch, (0, patch_size - patch.shape[2], 0, patch_size - patch.shape[1]), "constant", 0)
        
    #     return patch


    def extract_highest_contrast_patch(self, image):

        # Step 1: Compute the dark channels for both images
        dark_channel = torch.min(image, dim=1, keepdim=False)[0]

        # Step 2: Resize the dark channels to 8x8 images
        img_tensor = F.interpolate(dark_channel.unsqueeze(0), size=(8, 8), mode='bilinear', align_corners=False).squeeze(0)

        # Define Sobel kernels (make sure they are on the same device as the input tensor)
        sobel_horizontal_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to(image.device)
        sobel_vertical_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to(image.device)


        # Apply Sobel filters (for horizontal and vertical gradients)
        sobel_horizontal = F.conv2d(img_tensor.unsqueeze(0), weight=sobel_horizontal_kernel, stride=1, padding=1)
        sobel_vertical = F.conv2d(img_tensor.unsqueeze(0), weight=sobel_vertical_kernel, stride=1, padding=1)

        # Calculate the absolute values
        absx = sobel_horizontal.abs().squeeze()
        absy = sobel_vertical.abs().squeeze()

        # Combine the edges (same as cv2.addWeighted)
        edge = (absx + absy) / 2.0

        # Find the index of the highest intensity change
        # max_change_location = torch.unravel_index(edge.argmax(), edge.shape)

        max_loc_flat = torch.argmax(edge.view(-1))  # Flatten the 2D to 1D
        max_loc = divmod(max_loc_flat.item(), img_tensor.shape[1])  # (row, col) in 8x8 grid


        # Step 4: Project the locations to the original images' resolutions
        H, W = 256, 256
        scale_h = H / 8
        scale_w = W / 8
        max_loc_original = (max_loc[0] * scale_h, max_loc[1] * scale_w)

        x1_A, y1_A = int(max_loc_original[0]), int(max_loc_original[1])

        patch_size = 64

        # Ensure the crop is within bounds
        x1_A = min(x1_A, 192)  # Adjust x1 if it exceeds the width: 256 - 64
        y1_A = min(y1_A, 192)  # Adjust y1 if it exceeds the height: 256 - 64
        
        # Step 5: Crop a 64x64 patch from the image tensor
        # patch = image[:, :, y1_A:y1_A+patch_size, x1_A:x1_A+patch_size]
        
        # Step 6: Return these image patches
        return (x1_A, y1_A, x1_A+patch_size, y1_A+patch_size)


    def extract_darkest_patch(self, image):

        # Step 1: Compute the dark channels for both images
        dark_channel = torch.min(image, dim=1, keepdim=False)[0]

        # Step 2: Resize the dark channels to 8x8 images
        dark_channel_resized = F.interpolate(dark_channel.unsqueeze(0), size=(8, 8), mode='bilinear', align_corners=False).squeeze(0)

        # Step 3: Find the locations of the lowest values in the 8x8 resized dark channels
        min_loc_flat = torch.argmin(dark_channel_resized.view(-1))  # Flatten the 2D to 1D
        
        # Convert flat indices to 2D coordinates (row, col)
        min_loc = divmod(min_loc_flat.item(), dark_channel_resized.shape[1])  # (row, col) in 8x8 grid

        
        # Step 4: Project the locations to the original images' resolutions
        H, W = 256, 256
        scale_h = H / 8
        scale_w = W / 8
        min_loc_original = (min_loc[0] * scale_h, min_loc[1] * scale_w)

        x1_A, y1_A = int(min_loc_original[0]), int(min_loc_original[1])

        patch_size = 32

        # Ensure the crop is within bounds
        x1_A = min(x1_A, 192)  # Adjust x1 if it exceeds the width: 256 - 64
        y1_A = min(y1_A, 192)  # Adjust y1 if it exceeds the height: 256 - 64
        
        # Step 5: Crop a 64x64 patch from the image tensor
        patch = image[:, :, y1_A:y1_A+patch_size, x1_A:x1_A+patch_size]
        
        # Step 6: Return these image patches
        return patch


    # Patch-wise Histogram Contrastive Loss
    def calculate_PHC_loss(self, src, gen, tgt):

        # src: source image
        # gen: generated image
        # tgt: target image

        # num_bins = 100

        # patch1 = self.extract_darkest_patch(src)
        # patch2 = self.extract_darkest_patch(gen)
        # patch3 = self.extract_darkest_patch(tgt)

        # patch1 = self.extract_highest_contrast_patch(src)
        # patch2 = self.extract_highest_contrast_patch(gen)
        # patch3 = self.extract_highest_contrast_patch(tgt)

        x1, y1, x2, y2 = self.extract_highest_contrast_patch(src)
        patch1 = src[:, :, y1:y2, x1:x2]
        patch2 = gen[:, :, y1:y2, x1:x2]

        x1, y1, x2, y2 = self.extract_highest_contrast_patch(tgt)
        patch3 = tgt[:, :, y1:y2, x1:x2]

        # print(patch1.shape, patch2.shape, patch3.shape)

        # Define the coefficients for the luminance calculation
        coefficients = torch.tensor([0.2989, 0.5870, 0.1140], device=patch1.device)

        # Calculate luminance by performing a weighted sum of the RGB channels
        luminance1 = torch.sum(patch1 * coefficients.view(1, 3, 1, 1), dim=1, keepdim=True).squeeze(0)
        luminance2 = torch.sum(patch2 * coefficients.view(1, 3, 1, 1), dim=1, keepdim=True).squeeze(0)
        luminance3 = torch.sum(patch3 * coefficients.view(1, 3, 1, 1), dim=1, keepdim=True).squeeze(0)

        image_tensor1 = (luminance1 + 1) / 2
        image_tensor2 = (luminance2 + 1) / 2
        image_tensor3 = (luminance3 + 1) / 2

        # Flatten the image tensors to 1D arrays
        flat_image1 = image_tensor1.view(-1)
        flat_image2 = image_tensor2.view(-1)
        flat_image3 = image_tensor3.view(-1)

        # Compute histograms using PyTorch
        hist1 = torch.histc(flat_image1, bins=100, min=0.0, max=1.0)
        hist2 = torch.histc(flat_image2, bins=100, min=0.0, max=1.0)
        hist3 = torch.histc(flat_image3, bins=100, min=0.0, max=1.0)

        # Normalize histograms to get probability distributions
        hist1 = hist1 / (hist1.sum() + 0.00001)
        hist2 = hist2 / (hist2.sum() + 0.00001)
        hist3 = hist3 / (hist3.sum() + 0.00001)

        # Compute CDF (Cumulative Distribution Function) for both histograms
        cdf1 = torch.cumsum(hist1, dim=0)
        cdf2 = torch.cumsum(hist2, dim=0)
        cdf3 = torch.cumsum(hist3, dim=0)

        c = torch.nn.L1Loss()

        # phc_loss = (0.0001 * c(cdf2, cdf3)) / (c(cdf2, cdf1) + 0.00001)         # dehaze50
        phc_loss = (0.001 * c(cdf2, cdf3)) / (c(cdf2, cdf1) + 0.00001)         # dehaze51

        return phc_loss


    def extract_patches(self, image1, image2):

        # Step 1: Compute the dark channels for both images
        dark_channel1 = torch.min(image1, dim=1, keepdim=False)[0]
        dark_channel2 = torch.min(image2, dim=1, keepdim=False)[0]

        # Step 2: Resize the dark channels to 8x8 images
        dark_channel1_resized = F.interpolate(dark_channel1.unsqueeze(0), size=(8, 8), mode='bilinear', align_corners=False).squeeze(0)
        dark_channel2_resized = F.interpolate(dark_channel2.unsqueeze(0), size=(8, 8), mode='bilinear', align_corners=False).squeeze(0)

        # Step 3: Find the locations of the lowest values in the 8x8 resized dark channels
        min_loc1_flat = torch.argmin(dark_channel1_resized.view(-1))  # Flatten the 2D to 1D
        min_loc2_flat = torch.argmin(dark_channel2_resized.view(-1))  # Flatten the 2D to 1D
        
        # Convert flat indices to 2D coordinates (row, col)
        min_loc1 = divmod(min_loc1_flat.item(), dark_channel1_resized.shape[1])  # (row, col) in 8x8 grid
        min_loc2 = divmod(min_loc2_flat.item(), dark_channel2_resized.shape[1])  # (row, col) in 8x8 grid

        
        # Step 4: Project the locations to the original images' resolutions
        H, W = 256, 256
        scale_h = H / 8
        scale_w = W / 8
        min_loc1_original = (min_loc1[0] * scale_h, min_loc1[1] * scale_w)
        min_loc2_original = (min_loc2[0] * scale_h, min_loc2[1] * scale_w)

        x1_A, y1_A = int(min_loc1_original[0]), int(min_loc1_original[1])
        x1_B, y1_B = int(min_loc2_original[0]), int(min_loc2_original[1])

        patch_size = 64

        # Ensure the crop is within bounds
        x1_A = min(x1_A, 192)  # Adjust x1 if it exceeds the width: 256 - 64
        y1_A = min(y1_A, 192)  # Adjust y1 if it exceeds the height: 256 - 64
        x1_B = min(x1_B, 192)  # Adjust x1 if it exceeds the width: 256 - 64
        y1_B = min(y1_B, 192)  # Adjust y1 if it exceeds the height: 256 - 64
        # print(x1_A, y1_A, x1_B, y1_B)

        # print(image1.shape, image2.shape)
        
        # Step 5: Crop a 64x64 patch from the image tensor
        patch1 = image1[:, :, y1_A:y1_A+patch_size, x1_A:x1_A+patch_size]
        patch2 = image2[:, :, y1_B:y1_B+patch_size, x1_B:x1_B+patch_size]
        
        # Step 6: Return these image patches
        return patch1, patch2



    # # Patch-wise Histogram Alignment Loss
    # def calculate_PHA_loss(self, gen, tgt):

    #     # gen: generated image
    #     # tgt: target image

    #     # num_bins = 100

    #     # patch1, patch2 = self.extract_patches(gen, tgt)
    #     patch1 = self.extract_darkest_patch(gen)
    #     patch2 = self.extract_darkest_patch(tgt)

    #     # print(patch1.shape, patch2.shape)

    #     # Define the coefficients for the luminance calculation
    #     coefficients = torch.tensor([0.2989, 0.5870, 0.1140], device=patch1.device)

    #     # Calculate luminance by performing a weighted sum of the RGB channels
    #     luminance1 = torch.sum(patch1 * coefficients.view(1, 3, 1, 1), dim=1, keepdim=True).squeeze(0)
    #     luminance2 = torch.sum(patch2 * coefficients.view(1, 3, 1, 1), dim=1, keepdim=True).squeeze(0)

    #     # print(luminance1.shape, luminance2.shape)

    #     image_tensor1 = (luminance1 + 1) / 2
    #     image_tensor2 = (luminance2 + 1) / 2

    #     # # Flatten the image tensors to 1D arrays
    #     # flat_image1 = image_tensor1.view(-1).cpu().detach().numpy()
    #     # flat_image2 = image_tensor2.view(-1).cpu().detach().numpy()

    #     # # Compute histograms with 100 bins
    #     # hist1, bin_edges1 = np.histogram(flat_image1, bins=num_bins, range=(0, 1), density=True)
    #     # hist2, bin_edges2 = np.histogram(flat_image2, bins=num_bins, range=(0, 1), density=True)


    #     # # Compute CDF (Cumulative Distribution Function)
    #     # cdf1 = np.cumsum(hist1)  # CDF of the first histogram
    #     # cdf2 = np.cumsum(hist2)  # CDF of the second histogram

    #     # # Compute the Earth Mover's Distance (EMD)
    #     # emd = np.sum(np.abs(cdf1 - cdf2))/num_bins

    #     # dch_loss = torch.tensor(emd, device=image_tensor1.device)  # Ensure the tensor is on the same device

    #     # Flatten the image tensors to 1D arrays
    #     flat_image1 = image_tensor1.view(-1)
    #     flat_image2 = image_tensor2.view(-1)

    #     # Compute histograms using PyTorch
    #     hist1 = torch.histc(flat_image1, bins=100, min=0.0, max=1.0)
    #     hist2 = torch.histc(flat_image2, bins=100, min=0.0, max=1.0)

    #     # Normalize histograms to get probability distributions
    #     hist1 = hist1 / (hist1.sum() + 0.000001)
    #     hist2 = hist2 / (hist2.sum() + 0.000001)

    #     # Compute CDF (Cumulative Distribution Function) for both histograms
    #     cdf1 = torch.cumsum(hist1, dim=0)
    #     cdf2 = torch.cumsum(hist2, dim=0)

    #     # Compute the Earth Mover's Distance (EMD) by summing the absolute differences between the CDFs
    #     emd = torch.sum(torch.abs(cdf1 - cdf2)) / 100

    #     return emd




    # =========================================================================================================

    # def compute_differentiable_histogram(self, dark_channel, num_bins=256, epsilon=1e-6):
    #     """
    #     Compute a differentiable histogram for the dark channel.
    #     Args:
    #         dark_channel (Tensor): Dark channel tensor of shape (B, H, W).
    #         num_bins (int): Number of bins for the histogram.
    #         epsilon (float): Small constant for numerical stability.
    #     Returns:
    #         Tensor: Histogram tensor of shape (B, num_bins).
    #     """
    #     B, H, W = dark_channel.shape
    #     dark_channel = dark_channel.view(B, -1)  # Flatten to shape (B, H*W)
        
    #     # Create bin centers
    #     bin_edges = torch.linspace(-1.0, 1.0, steps=num_bins, device=dark_channel.device)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Midpoints of bins
        
    #     # Compute soft assignments to bins
    #     dark_channel = dark_channel.unsqueeze(-1)  # Shape: (B, H*W, 1)
    #     bin_centers = bin_centers.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_bins)
    #     weights = torch.exp(-((dark_channel - bin_centers) ** 2) / epsilon)  # Gaussian kernel
    #     weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)  # Normalize
        
    #     # Sum weights to form histograms
    #     histograms = weights.sum(dim=1)  # Shape: (B, num_bins)

    #     # Normalize histograms
    #     histograms /= (H * W)


    #     # Multiply epsilon to the final loss value for normalization:
    #     # histograms *= epsilon

    #     return histograms


    # def compute_wasserstein_distance(self, hist1, hist2):
    #     """
    #     Compute the Wasserstein distance between two histograms.
    #     Args:
    #         hist1 (Tensor): Histogram tensor of shape (B, num_bins).
    #         hist2 (Tensor): Histogram tensor of shape (B, num_bins).
    #     Returns:
    #         Tensor: Wasserstein distance for each batch element.
    #     """
    #     # Compute cumulative distributions (CDFs)
    #     cdf1 = torch.cumsum(hist1, dim=1)
    #     cdf2 = torch.cumsum(hist2, dim=1)
    #     # Compute the Wasserstein distance
    #     wasserstein_distance = torch.sum(torch.abs(cdf1 - cdf2), dim=1)
    #     return wasserstein_distance


    # # Dark Channel Histogram Loss
    # def calculate_DCH_loss(self, gen, tgt):

    #     # gen: generated image
    #     # tgt: target image

    #     # Compute dark channels
    #     gen_dc = torch.min(gen, dim=1)[0]  # Shape: (B, H, W)
    #     tgt_dc = torch.min(tgt, dim=1)[0]  # Shape: (B, H, W)

    #     # Compute histograms
    #     # gen_dc_hist = self.compute_histogram(gen_dc)
    #     # tgt_dc_hist = self.compute_histogram(tgt_dc)
    #     gen_dc_hist = self.compute_differentiable_histogram(gen_dc)
    #     tgt_dc_hist = self.compute_differentiable_histogram(tgt_dc)

    #     # Compute Wasserstein distance
    #     dch_loss = self.compute_wasserstein_distance(gen_dc_hist, tgt_dc_hist).mean()

    #     return dch_loss

    # =========================================================================================================