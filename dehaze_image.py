import os
import torch
import util.util as util
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from options.test_options import TestOptions
from models.networks import KATGenerator
torch.cuda.empty_cache()


# Model configuration ('s': small, 'b': base)
modeltype = 's'
opt = TestOptions().parse()  # get test options
if modeltype == 's':        
    opt.ngf = 32
    opt.n_kat_blocks = 9
    ckpt_path = 'checkpoints/uid_kat_s.pth'     # UID-KAT-S
elif modeltype == 'b':       
    opt.ngf = 64
    opt.n_kat_blocks = 5
    ckpt_path = 'checkpoints/uid_kat_b.pth'     # UID-KAT-B


# Input/output configuration
image_path = 'datasets/test1/0030.jpg'
save_path = f'datasets/test1/0030_uidkat_{modeltype}.png'


# Main
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to a tensor with values in [0, 1]
    ])
image_tensor = transform(image)
image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
C, H, W = image_tensor.shape

patch_size = 4
factor = patch_size*4     # due to 2 times of downscaling
testsize = ((max(H, W) + factor - 1) // factor) * factor

image_tensor = image_tensor.unsqueeze(0)
image_tensor = F.interpolate(image_tensor, size=(testsize, testsize), mode='bicubic', align_corners=False)
# print(image_tensor.shape)

model = KATGenerator(input_nc=3, output_nc=3, opt=opt)
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint)
model.eval()

# Move the tensor to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_tensor = image_tensor.to(device)

# Get the model's prediction
with torch.no_grad():  # No need to track gradients during inference
    output = model(image_tensor)  # Model output (logits for classification)

output = F.interpolate(output, size=(H, W), mode='bicubic', align_corners=False)
im = util.tensor2im(output)
util.save_image(im, save_path, aspect_ratio=1.0)
print('Done!')