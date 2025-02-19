import os
import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from tqdm import tqdm
import torch.nn.functional as F


import torch
torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)


def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


def pad_to_devisible_by_patch_size(x, patch_size):
    """
    Pads an input tensor to make it square with a size divisible by 32.

    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W].

    Returns:
        torch.Tensor: Padded tensor with shape [B, C, N, N].
        tuple: Original (H, W) dimensions for unpadding.
    """
    B, C, H, W = x.shape

    # Calculate target size N (smallest multiple of 32 >= max(H, W))

    factor = patch_size*4     # due to 2 times of downscaling

    # N = ((max(H, W) + 31) // 32) * 32
    N = ((max(H, W) + factor - 1) // factor) * factor

    # Calculate padding
    pad_h = N - H  # Total padding needed for height
    pad_w = N - W  # Total padding needed for width

    # Divide padding into top/bottom and left/right
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)

    return x_padded, (H, W)



def unpad_to_original(x_padded, original_shape):
    """
    Removes padding from a tensor to restore its original shape.

    Args:
        x_padded (torch.Tensor): Padded tensor with shape [B, C, N, N].
        original_shape (tuple): Original (H, W) dimensions.

    Returns:
        torch.Tensor: Tensor cropped to the original shape [B, C, H, W].
    """
    H, W = original_shape  # Original height and width

    _, _, pH, pW = x_padded.shape

    h1 = (pH - H)//2
    h2 = (pH + H)//2
    w1 = (pW - W)//2
    w2 = (pW + W)//2

    # Crop the tensor back to the original dimensions
    x_original = x_padded[:, :, h1:h2, w1:w2]
    return x_original



if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)      # create a model given opt.model and other options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # print(model)

    output_dir = opt.results_dir
    test_mode = opt.input_mode
    patch_size = opt.patch_size

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # testsize = 512
    # test_mode = 'padding'     # 'resize' or 'padding'

    total_time = 0

    for i, data in enumerate(tqdm(dataset)):

        realA = data['A']

        if test_mode == 'resize':
            B, C, H, W = realA.shape

            if H > 2000 or W > 2000:
                H = W = 512

            factor = patch_size*4     # due to 2 times of downscaling
            testsize = ((max(H, W) + factor - 1) // factor) * factor

            # realA_resized = F.interpolate(realA, size=(testsize, testsize), mode='bilinear', align_corners=False)
            realA_resized = F.interpolate(realA, size=(testsize, testsize), mode='bicubic', align_corners=False)
            data['A'] = realA_resized
        elif test_mode == 'padding':
            # realA_padded, (H,W) = pad_to_square_32(realA)
            realA_padded, (H,W) = pad_to_devisible_by_patch_size(realA, patch_size)
            data['A'] = realA_padded
        else:
            print('Test mode is not defined!')
            exit()        
        
        # Model configuration...

        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()

        # exit() # <=====================


        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break


        start = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        end = time.time()

        if i == 0:
            # Ignore the first time
            proc_time = 0
        else:
            proc_time = end - start

        total_time += proc_time

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        # print(f'Processing {img_path[0]} | Time: {proc_time}')

        fname = get_file_name(img_path[0])

        for label, im_data in visuals.items():

            if label != 'fake_B': continue

            if test_mode == 'resize':
                # im_data = F.interpolate(im_data, size=(H, W), mode='bilinear', align_corners=False)
                im_data = F.interpolate(im_data, size=(H, W), mode='bicubic', align_corners=False)
            elif test_mode == 'padding':
                im_data = unpad_to_original(im_data, (H, W))

            # print('Min/max before:', im_data.min(), im_data.max())
            im = util.tensor2im(im_data)
            # print('Min/max after:', im.min(), im.max())

            save_path = f'{output_dir}/{fname}.png'
            util.save_image(im, save_path, aspect_ratio=1.0)

    print(f'Average processing time: {total_time/(len(dataset)-1)} seconds/image.')