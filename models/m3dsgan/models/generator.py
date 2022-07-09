import torch.nn as nn
import torch.nn.functional as F
import torch
from models.common import (
    arange_pixels, image_points_to_world, origin_to_world
)
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from models.camera import get_camera_mat, get_random_pose, get_camera_pose

class Generator(nn.Module):
    ''' 3dsgan Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        decoder (nn.Module): decoder network
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        background_generator (nn.Module): background generator
        bounding_box_generaor (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
        background_rotation_range (tuple): background rotation range
         (0 - 1)
        sample_object-existance (bool): whether to sample the existance
            of objects; only used for clevr2345
        use_max_composition (bool): whether to use the max
            composition operator instead
    '''

    def __init__(self, device, z_dim=256, decoder=None,
                 range_u=(0, 0), range_v=(0.25, 0.25), n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 background_generator=None,
                 bounding_box_generator=None, resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 backround_rotation_range=[0., 0.],
                 sample_object_existance=False,
                 use_max_composition=False, app_dim=64, eigen_learning=False, **kwargs):

        super().__init__()

        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.bounding_box_generator = bounding_box_generator
        self.fov = fov
        self.backround_rotation_range = backround_rotation_range
        self.sample_object_existance = sample_object_existance
        self.z_dim = z_dim
        self.n_basic = 3
        self.use_max_composition = use_max_composition
        self.app_dim = app_dim
        self.camera_matrix = get_camera_mat(fov=fov).to(device)
        self.eigen_learning = eigen_learning

        if self.eigen_learning:
            self.projection = SubspaceLayer(dim=z_dim, n_basis=self.n_basic)
        self.fixed_z = None
        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None
        if bounding_box_generator is not None:
            self.bounding_box_generator = bounding_box_generator.to(device)
        else:
            self.bounding_box_generator = bounding_box_generator
        if neural_renderer is not None:
            self.neural_renderer = neural_renderer.to(device)
        else:
            self.neural_renderer = None

    def forward(self, batch_size=32, latent_codes=None, camera_matrices=None,
                transformations=None, mode="training", it=0,
                return_alpha_map=False,
                not_render_background=True,
                only_render_background=False):

        if latent_codes is None:
            latent_codes = self.get_latent_codes(batch_size)

        if camera_matrices is None:
            camera_matrices = self.get_random_camera(batch_size)

        if transformations is None:
            transformations = self.get_random_transformations(batch_size)

        if return_alpha_map:
            rgb_v, alpha_map = self.volume_render_image(
                latent_codes, camera_matrices, transformations,
                mode=mode, it=it, return_alpha_map=True,
                not_render_background=not_render_background)

            if self.neural_renderer is not None:
                rgb = self.neural_renderer(alpha_map)
            else:
                rgb = rgb_v
            return rgb, alpha_map

        else:
            rgb_v = self.volume_render_image(
                latent_codes, camera_matrices, transformations,
                mode=mode, it=it, not_render_background=not_render_background,
                only_render_background=only_render_background)
            if self.neural_renderer is not None:
                rgb = self.neural_renderer(rgb_v)
            else:
                rgb = rgb_v
            return rgb

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).sum()
                )
        return sum(reg) / len(reg)

    def get_latent_codes(self, batch_size=32, tmp=1.):

        z_sem_obj = self.sample_z((batch_size, self.z_dim),  tmp=tmp, is_canonical=False)
        z_pose_obj = self.sample_z((batch_size, self.z_dim),  tmp=tmp, is_canonical=False)

        return z_sem_obj, z_pose_obj

    def get_latent_codes_(self, batch_size=32, tmp=1.):

        z_sem1 = self.sample_z((batch_size, self.z_dim),  tmp=tmp, is_canonical=False)
        z_sem2 = self.sample_z((batch_size, self.z_dim),  tmp=tmp, is_canonical=False)
        z_pose1 = self.sample_z((batch_size, self.z_dim),  tmp=tmp, is_canonical=False)
        z_pose2 = self.sample_z((batch_size, self.z_dim),  tmp=tmp, is_canonical=False)

        return z_sem1, z_sem2, z_pose1, z_pose2

    def sample_z(self, size, to_device=True, tmp=1., is_canonical=False):
        if is_canonical:
            z = torch.ones(*size) * tmp
            if to_device:
                z = z.to(self.device)
            return z
        else:
            z = torch.randn(*size) * tmp
            if to_device:
                z = z.to(self.device)
            return z

    def get_vis_dict(self, batch_size=64):
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': self.get_latent_codes(batch_size),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
        }
        return vis_dict

    # for inverse
    def get_vis_dict_inverse(self, batch_size=64):
        semcode, posecode = self.get_latent_codes(batch_size)
        s, t, r = self.get_random_transformations(batch_size)
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': (semcode, posecode),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
        }
        return vis_dict, semcode, posecode, s, t, r

    def get_vis_dict_(self, batch_size=64):
        sem1, sem2, pose1, pose2 = self.get_latent_codes_(batch_size)
        vis_dict1 = {
            'batch_size': batch_size,
            'latent_codes': (sem1, pose1),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
        }
        vis_dict2 = {
            'batch_size': batch_size,
            'latent_codes': (sem1, pose2),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
        }
        vis_dict3 = {
            'batch_size': batch_size,
            'latent_codes': (sem2, pose1),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
        }
        return vis_dict1, vis_dict2, vis_dict3

    def get_random_camera(self, batch_size=32, to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_random_pose(
            self.range_u, self.range_v, self.range_radius, batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_camera(self, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32,
                   to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_camera_pose(
            self.range_u, self.range_v, self.range_radius, val_u, val_v,
            val_r, batch_size=batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_random_bg_rotation(self, batch_size, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(Rot.from_euler(
                    'z', r_random * 2 * np.pi).as_dcm()
                ) for i in range(batch_size)]
            R_bg = torch.stack(R_bg, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            R_bg = R_bg.to(self.device)
        return R_bg
    def get_bg_rotation(self, val, batch_size=32, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
            r = torch.from_numpy(
                Rot.from_euler('z', r_val * 2 * np.pi).as_dcm()
            ).reshape(1, 3, 3).repeat(batch_size, 1, 1).float()
        else:
            r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            r = r.to(self.device)
        return r

    def get_random_transformations(self, batch_size=32, to_device=True):
        device = self.device
        s, t, R, _ = self.bounding_box_generator(batch_size)
        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_random_transformations_(self, batch_size=32, to_device=True):
        device = self.device
        s, t, R, Rv2 = self.bounding_box_generator(batch_size)
        if to_device:
            s, t, R, Rv2 = s.to(device), t.to(device), R.to(device), Rv2.to(device)
        return s, t, R, Rv2

    def get_transformations(self, val_s=[[0.5, 0.5, 0.5]],
                            val_t=[[0.5, 0.5, 0.5]], val_r=[0.5],
                            batch_size=32, to_device=True):
        device = self.device
        s = self.bounding_box_generator.get_scale(
            batch_size=batch_size, val=val_s)
        t = self.bounding_box_generator.get_translation(
            batch_size=batch_size, val=val_t)
        R = self.bounding_box_generator.get_rotation(
            batch_size=batch_size, val=val_r)

        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations_in_range(self, range_s=[0., 1.], range_t=[0., 1.],
                                     range_r=[0., 1.], n_boxes=1,
                                     batch_size=32, to_device=True):
        s, t, R = [], [], []

        def rand_s(): return range_s[0] + \
            np.random.rand() * (range_s[1] - range_s[0])

        def rand_t(): return range_t[0] + \
            np.random.rand() * (range_t[1] - range_t[0])
        def rand_r(): return range_r[0] + \
            np.random.rand() * (range_r[1] - range_r[0])

        for i in range(batch_size):
            val_s = [[rand_s(), rand_s(), rand_s()] for j in range(n_boxes)]
            val_t = [[rand_t(), rand_t(), rand_t()] for j in range(n_boxes)]
            val_r = [rand_r() for j in range(n_boxes)]
            si, ti, Ri = self.get_transformations(
                val_s, val_t, val_r, batch_size=1, to_device=to_device)
            s.append(si)
            t.append(ti)
            R.append(Ri)
        s, t, R = torch.cat(s), torch.cat(t), torch.cat(R)
        if to_device:
            device = self.device
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_rotation(self, val_r, batch_size=32, to_device=True):
        device = self.device
        R = self.bounding_box_generator.get_rotation(
            batch_size=batch_size, val=val_r)
        if to_device:
            R = R.to(device)
        return R

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx=0,
                                scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)
                                     ).permute(0, 2, 1)).permute(
            0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box

    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 rotation_matrix):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        camera_world = (rotation_matrix @
                        camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        pixels_world = (rotation_matrix @
                        pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
        ray_world = pixels_world - camera_world

        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_world.unsqueeze(-2).contiguous()
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations):
        batch_size = pixels_world.shape[0]
        # n_steps = 64
        n_steps = di.shape[-1]
        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations)
        ray_i = pixels_world_i - camera_world_i
        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()

        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)

        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(
                                         1, -1, 1), torch.arange(ns).reshape(
                                             1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)
        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def get_object_existance(self, n_boxes, batch_size=32):
        '''
        Note: We only use this setting for Clevr2345, so that we can hard-code
        the probabilties here. If you want to apply it to a different scenario,
        you would need to change these.
        '''
        probs = [
            .19456788355146545395,
            .24355003312266127155,
            .25269546846185522711,
            .30918661486401804737,
        ]

        n_objects_prob = np.random.rand(batch_size)
        n_objects = np.zeros_like(n_objects_prob).astype(np.int)
        p_cum = 0
        obj_n = [i for i in range(2, n_boxes + 1)]
        for idx_p in range(len(probs)):
            n_objects[
                (n_objects_prob >= p_cum) &
                (n_objects_prob < p_cum + probs[idx_p])
            ] = obj_n[idx_p]
            p_cum = p_cum + probs[idx_p]
            assert(p_cum <= 1.)

        object_existance = np.zeros((batch_size, n_boxes))
        for b_idx in range(batch_size):
            n_obj = n_objects[b_idx]
            if n_obj > 0:
                idx_true = np.random.choice(
                    n_boxes, size=(n_obj,), replace=False)
                object_existance[b_idx, idx_true] = True
        object_existance = object_existance.astype(np.bool)
        return object_existance

    def volume_render_image(self, latent_codes, camera_matrices,
                            transformations, mode='training',
                            it=0, return_alpha_map=False,
                            not_render_background=False,
                            only_render_background=False):
        res = self.resolution_vol
        device = self.device
        n_steps = self.n_ray_samples
        n_points = res * res
        depth_range = self.depth_range
        batch_size = latent_codes[0].shape[0]
        z_sem_obj, z_pose_obj = latent_codes
        assert(not (not_render_background and only_render_background))

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size,
                               invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.

        # Project to 3D world
        pixels_world = image_points_to_world(
            pixels, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        camera_world = origin_to_world(
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
            torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        if mode == 'training':
            di = self.add_noise_to_interval(di)

        p_, r_ = self.get_evaluation_points(
            pixels_world, camera_world, di, transformations)

        z_sem_ , z_pose = z_sem_obj, z_pose_obj
        feat_, sigma_ = self.decoder(p_, r_, z_sem_, z_pose)

        if mode == 'training':
            # As done in NeRF, add noise during training
            sigma_ += torch.randn_like(sigma_)

        # Mask out values outside
        padd = 0.1
        mask_box = torch.all(
            p_ <= 1. + padd, dim=-1) & torch.all(
                p_ >= -1. - padd, dim=-1)
        sigma_[mask_box == 0] = 0.
        sigma_ = sigma_.reshape(batch_size, n_points, n_steps)
        sigma_ = F.relu(sigma_)

        feat_ = feat_.reshape(batch_size, n_points, n_steps, -1)

        #feat_ = torch.softmax(feat_,dim=-1)
        weights = self.calc_volume_weights(di, ray_vector, sigma_)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_, dim=-2)

        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, res, res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y

        rgb = feat_map
        if return_alpha_map:
            weights_obj = self.calc_volume_weights(
                di, ray_vector, sigma_, last_dist=0.)
            acc_map = torch.sum(weights_obj, dim=-1, keepdim=True)
            acc_map = acc_map.permute(0, 2, 1).reshape(
                batch_size, -1, res, res)
            acc_map = acc_map.permute(0, 1, 3, 2)
            return rgb, acc_map
        else:
            return rgb

class SubspaceLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_basis: int,
    ):
        super().__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu

