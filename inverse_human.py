import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from models import config
from torchvision import transforms, utils
import logging
from torch.nn import functional as F
from models.Losses.VGG import VGGLoss
from models.utils import tensor2label
import cv2
import lpips

logger_py = logging.getLogger(__name__)
# np.random.seed(0)
# torch.manual_seed(0)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D-SGAN model.'
)

parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument('--save_path', type=str, default='./out/fashion/',
                    help='Path for saving inversing images')
parser.add_argument('--epoches', type=int, default=20000,
                    help='overall epoches')
parser.add_argument('--lambda_l1', type=float, default=1.0,
                    help='overall epoches')
parser.add_argument('--lambda_vggc', type=float, default=0.0,
                    help='overall epoches')
parser.add_argument('--lambda_vggs', type=float, default=0.0,
                    help='overall epoches')
parser.add_argument('--lambda_lpips', type=float, default=1,
                    help='overall epoches')
parser.add_argument('--solver', type=str, default='adam',
                    help='which solver')
parser.add_argument('--id', type=int, default=40,
                    help='overall epoches')
args = parser.parse_args()
args.save_path = os.path.join(args.save_path, 'inverse_{}_real'.format(args.id))

print(args.save_path)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['inverse']['out_dir']
backup_every = cfg['inverse']['backup_every']
exit_after = args.exit_after
lr = cfg['inverse']['learning_rate']
lr_d = cfg['inverse']['learning_rate_d']
batch_size = cfg['inverse']['batch_size']
n_workers = cfg['inverse']['n_workers']

t0 = time.time()

def toImg(x, isfake=False):

    if isfake:
        x = (x.cpu().detach().numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5
    else:
        x = x.cpu().detach().numpy().transpose(0,2,3,1) * 255.0
    x = x[0][:,:,::-1]
    return x

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

test_dataset = config.get_dataset(cfg)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=n_workers, shuffle=False,
    pin_memory=True, drop_last=True,
)

def L1_loss(real, fake):
    return F.l1_loss(real, fake)

model = config.get_model(cfg, device=device, len_dataset=len(test_dataset))
# Initialize training

opti = optim.Adam

g = model.seg2imgnet
g_eval = g
g_eval.eval()

for p in g.parameters():
    p.requires_grad_(False)

sample_z = torch.randn(1, 512, device=device)
sample_z.requires_grad = True

optimizer = opti({sample_z}, lr=0.1, betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
vggloss = VGGLoss().to(device)
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

data = list(test_loader)[args.id]

real_img = data['image']
real_img_seg = data['seg']

real_img = real_img.to(device)
real_img_seg = real_img_seg.to(device)

epoch_it = 0.0
t0b = time.time()
for i in range(args.epoches):

    print(epoch_it)

    epoch_it += 1
    optimizer.zero_grad()

    fake_img = g(real_img_seg, sample_z, None, is_sampling=True)
    fake_img = (fake_img + 1.0) / 2.0

    l1_loss = L1_loss(real_img, fake_img)
    lpips_loss = loss_fn_alex(real_img, fake_img)

    overall_g_loss = args.lambda_l1 * l1_loss + args.lambda_lpips * lpips_loss

    g.zero_grad()
    overall_g_loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch_it % 100 == 0:
        print('lr', scheduler.get_last_lr(), 'epoch_it', epoch_it, 'overall loss', overall_g_loss.item(), 'l1_loss', l1_loss.item(), 'lpips_loss', lpips_loss.item())

    if epoch_it % 100 == 0:

        out_file_name = 'output{}.jpg'.format(epoch_it)
        out_file_name_fake = 'output_fake{}.jpg'.format(epoch_it)
        real_img_seg_ = tensor2label(real_img_seg, 8)
        real_img_ = toImg(real_img)
        fake_img_ = toImg(fake_img)
        img_ = np.stack((real_img_, real_img_seg_, fake_img_), axis=0)
        # real and fake
        img_ = np.transpose(img_, (1,0,2,3))
        h,b,w,c = img_.shape
        img_ = img_.reshape((h,b*w,c))
        cv2.imwrite(os.path.join(args.save_path, out_file_name), img_)
        utils.save_image(fake_img, os.path.join(args.save_path, out_file_name_fake))

# for different seg as input
appnp_path = './out/fashion256/inverse_{}/app/'.format(args.id)
for i in range(64):

    img = np.load(os.path.join(appnp_path, 'app_{}.npy'.format(i)))
    img = torch.from_numpy(img).to(device)
    img = img.unsqueeze(0)

    out_file_name = 'app_{}.jpg'.format(i)
    out_file_name_seg = 'app_{}seg.jpg'.format(i)
    app_path = os.path.join(args.save_path, 'app')
    if not os.path.exists(app_path):
        os.mkdir(app_path)
    x_fake = g_eval(img, sample_z, None, is_sampling=True)
    x_fake = toImg(x_fake, isfake=True)

    cv2.imwrite(os.path.join(app_path, out_file_name), x_fake)

posenp_path = './out/fashion256/inverse_{}/pose/'.format(args.id)
for i in range(64):

    img = np.load(os.path.join(posenp_path, 'pose_{}.npy'.format(i)))
    img = torch.from_numpy(img).to(device)
    img = img.unsqueeze(0)

    out_file_name = 'pose_{}.jpg'.format(i)
    pose_path = os.path.join(args.save_path, 'pose')
    if not os.path.exists(pose_path):
        os.mkdir(pose_path)
    x_fake = g_eval(img, sample_z, None, is_sampling=True)
    x_fake = toImg(x_fake, isfake=True)
    cv2.imwrite(os.path.join(pose_path, out_file_name), x_fake)

rnp_path = './out/fashion256/inverse_{}/rotation/'.format(args.id)
for i in range(64):

    img = np.load(os.path.join(rnp_path, 'R_{}.npy'.format(i)))
    img = torch.from_numpy(img).to(device)
    img = img.unsqueeze(0)

    out_file_name = 'R_{}.jpg'.format(i)
    rotation = os.path.join(args.save_path, 'rotation')
    if not os.path.exists(rotation):
        os.mkdir(rotation)
    x_fake = g_eval(img, sample_z, None, is_sampling=True)
    x_fake = toImg(x_fake, isfake=True)
    cv2.imwrite(os.path.join(rotation, out_file_name), x_fake)