import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from models import config
import logging
from torch.nn import functional as F
from models.Losses.VGG import VGGLoss
from models.utils import tensor2label
import cv2
from torchvision.utils import save_image, make_grid

logger_py = logging.getLogger(__name__)
np.random.seed(0)
torch.manual_seed(0)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a mm3dsgan model.'
)

parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument('--save_path', type=str, default='./out/fashion/inverse_40',
                    help='Path for saving inversing images')
parser.add_argument('--epoches', type=int, default=20000,
                    help='overall epoches')
parser.add_argument('--id', type=int, default=40,
                    help='overall epoches')
args = parser.parse_args()

def toImg(x):
    x = x.cpu().detach().numpy().transpose(0,2,3,1) * 255.0
    x = x[0][:,:,::-1]
    return x

def toNpy(x):
    x = x.cpu().detach().numpy()[0]
    return x

def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z

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

g = model.generator
g_eval = g
g_eval.eval()

for p in g.parameters():
    p.requires_grad_(False)

latent_dict, app_, pose_, s, t, r = g.get_vis_dict_inverse(1)

app_.requires_grad = True
pose_.requires_grad = True
s.requires_grad = True
t.requires_grad = True
r.requires_grad = True

optimizer = opti({app_, pose_, s, t, r}, lr=0.1, betas=(0.9, 0.999), eps=1e-8)

try:
    ckpt = torch.load('./out/fashion256/model.pt', map_location='cpu')
    new_dict_ = model.state_dict()
    for k, v in ckpt['model'].items():
        if k in new_dict_:
            new_dict_[k] = v
    model.load_state_dict(new_dict_)
    print("Loaded model checkpoint.")
except FileExistsError:
    load_dict = dict()
    print("No model checkpoint found.")

vggloss = VGGLoss().to(device)
data = list(test_loader)[args.id]

epoch_it = 0.0
t0b = time.time()

seg = data['seg']
real_img = seg.to(device)

for i in range(args.epoches):
    epoch_it += 1

    optimizer.zero_grad()
    x_fake_seg = g(**latent_dict)

    l1_loss = L1_loss(real_img, x_fake_seg)
    overall_g_loss = l1_loss

    g.zero_grad()
    overall_g_loss.backward()
    optimizer.step()

    if epoch_it % 100 == 0:
        print('epoch', epoch_it, 'overall loss', overall_g_loss.item(), 'l1_loss', l1_loss.item())

    if epoch_it % 100 == 0:

        out_file_name = 'output{}.jpg'.format(epoch_it)
        out_file_name_seg = 'output{}_seg.jpg'.format(epoch_it)

        fakeimage = tensor2label(x_fake_seg, 8, imtype=np.float32, tile=True, forsave=True)
        save_image(torch.from_numpy(fakeimage), os.path.join(args.save_path, out_file_name_seg))

        fakeimage = (fakeimage.transpose(0,2,3,1)[0]) * 255.0
        realimage = tensor2label(real_img, 8, imtype=np.float32, tile=True, forsave=True)
        realimage = (realimage.transpose(0,2,3,1)[0]) * 255.0

        img_ = np.stack((fakeimage, realimage), axis=0)
        # real and fake
        img_ = np.transpose(img_, (1,0,2,3))
        h,b,w,c = img_.shape
        img_ = img_.reshape((h,b*w,c))

        cv2.imwrite(os.path.join(args.save_path, out_file_name), img_)

# for different app_code
app_code = g_eval.get_latent_codes(1)[0]

for i in range(64):

    out_file_name = 'app_{}.jpg'.format(i)
    out_file_name_npy = 'app_{}.npy'.format(i)
    app_path = os.path.join(args.save_path, 'app')

    if not os.path.exists(app_path):
        os.mkdir(app_path)

    w = i * 1.0 / ((64) - 1)
    z_ii = interpolate_sphere(app_code, app_, w)
    latent_dict_ = latent_dict.copy()
    latent_dict_['latent_codes'] = [z_ii, pose_]
    x_fake_seg = g_eval(**latent_dict_)
    fakeimage = tensor2label(x_fake_seg, 8, imtype=np.float32, tile=True, forsave=True)
    save_image(torch.from_numpy(fakeimage), os.path.join(app_path, out_file_name))
    fakeimage = (fakeimage.transpose(0, 2, 3, 1)[0] + 1.0) * 127.5
    # cv2.imwrite(os.path.join(app_path, out_file_name), fakeimage)
    np.save(os.path.join(app_path, out_file_name_npy), toNpy(x_fake_seg))

# for different pose code
app_pose_code = g_eval.get_latent_codes(1)[1]
app_pose_codev2 = g_eval.get_latent_codes(1)[1]

for i in range(64):

    out_file_name = 'pose_{}.jpg'.format(i)
    out_file_name_npy = 'pose_{}.npy'.format(i)

    app_path = os.path.join(args.save_path, 'pose')
    if not os.path.exists(app_path):
        os.mkdir(app_path)

    w = i * 1.0 / ((64) - 1)
    z_ii = interpolate_sphere(app_pose_code, app_pose_codev2, w)

    latent_dict_ = latent_dict.copy()
    latent_dict_['latent_codes'] = [app_, z_ii]
    x_fake_seg = g_eval(**latent_dict_)

    fakeimage = tensor2label(x_fake_seg, 8, imtype=np.float32, tile=True, forsave=True)
    save_image(torch.from_numpy(fakeimage), os.path.join(app_path, out_file_name))
    fakeimage = (fakeimage.transpose(0, 2, 3, 1)[0] + 1.0) * 127.5
    np.save(os.path.join(app_path, out_file_name_npy), toNpy(x_fake_seg))

# for different T
n_steps = 64
r_scale = [0., 1.]

print(latent_dict['latent_codes'][0][0][0])

for i in range(n_steps):
    out_file_name = 'R_{}.jpg'.format(i)
    out_file_name_npy = 'R_{}.npy'.format(i)
    app_path = os.path.join(args.save_path, 'rotation')

    if not os.path.exists(app_path):
        os.mkdir(app_path)

    r_ = [i * 1.0 / (n_steps - 1) for j in range(1)]
    r_ = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r_]
    r_ = g_eval.get_rotation(r_, 1)

    latent_dict_ = latent_dict.copy()
    latent_dict_['transformations'] = [s, t, r_]
    x_fake_seg = g(**latent_dict_)
    fakeimage = tensor2label(x_fake_seg, 8, imtype=np.float32, tile=True, forsave=True)
    save_image(torch.from_numpy(fakeimage), os.path.join(app_path, out_file_name))
    fakeimage = (fakeimage.transpose(0, 2, 3, 1)[0] + 1.0) * 127.5
    #cv2.imwrite(os.path.join(app_path, out_file_name), fakeimage)
    np.save(os.path.join(app_path, out_file_name_npy), toNpy(x_fake_seg))

out_file_name = 'img.jpg'
out_file_name_seg = 'seg.jpg'
real_path = os.path.join(args.save_path, 'real')

if not os.path.exists(real_path):
    os.mkdir(real_path)

segimage = tensor2label(seg, 8, imtype=np.float32, tile=True, forsave=True)
save_image(torch.from_numpy(segimage), os.path.join(real_path, out_file_name_seg))

image = data['image']
image_ = toImg(image)
cv2.imwrite(os.path.join(real_path, out_file_name), image_)







