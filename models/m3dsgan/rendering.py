import torch
import numpy as np
from models.common import interpolate_sphere
from torchvision.utils import save_image, make_grid
import imageio
from math import sqrt
from os import makedirs
from os.path import join
from models.utils import tensor2label

class Renderer(object):
    '''  Render class for mm3dsgan.
    It provides functions to render the representation.
    Args:
        model (nn.Module): trained mm3dsgan model
        device (device): pytorch device
    '''

    def __init__(self, model, device=None):
        self.model = model.to(device)
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        self.generator = gen
        # sample temperature; only used for visualiations
        self.sample_tmp = 1.0

        seg2imgnet = self.model.seg2imgnet
        if seg2imgnet is None:
            seg2imgnet = self.model.seg2imgnet

        seg2imgnet.eval()
        self.seg2imgnet = seg2imgnet

    def set_random_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def render_full_visualization(self, img_out_path,
                                  render_program=['object_rotation']):
        for rp in render_program:
            if rp == 'object_rotation':
                self.set_random_seed()
                self.render_object_rotation(img_out_path)
            if rp == 'random_samples':
                self.set_random_seed()
                self.render_object_random(img_out_path)
            if rp == 'object_rotation_random':
                self.set_random_seed()
                self.object_rotation_random(img_out_path)
            if rp == 'object_translation_horizontal':
                self.set_random_seed()
                self.render_object_translation_horizontal(img_out_path, mode='t1')
            if rp == 'object_translation_horizontalv2':
                self.set_random_seed()
                self.render_object_translation_horizontal(img_out_path, mode='t2')
            if rp == 'object_translation_horizontalv3':
                self.set_random_seed()
                self.render_object_translation_horizontal(img_out_path, mode='t3')
            if rp == 'object_translation_vertical':
                self.set_random_seed()
                self.render_object_translation_depth(img_out_path)
            if rp == 'interpolate_app':
                self.set_random_seed()
                self.render_interpolation(img_out_path)
            if rp == 'interpolate_as':
                self.set_random_seed()
                self.render_interpolation_as(img_out_path)
            if rp == 'interpolate_pose':
                self.set_random_seed()
                self.render_interpolation(img_out_path, mode='pose', n_steps=32)
            if rp == 'interpolate_app_random':
                self.set_random_seed()
                self.render_interpolation_random(img_out_path)
            if rp == 'interpolate_pose_random':
                self.set_random_seed()
                self.render_interpolation_random(img_out_path, mode='pose')
            if rp == 'interpolate_shape':
                self.set_random_seed()
                self.render_interpolation(img_out_path, mode='shape')
            if rp == 'render_camera_elevation':
                self.set_random_seed()
                self.render_camera_elevation(img_out_path)

    def object_rotation_random(self, img_out_path, batch_size=1, n_steps=5000):

        gen = self.generator
        seg2imgnet = self.seg2imgnet

        bbox_generator = gen.bounding_box_generator
        n_boxes = bbox_generator.n_boxes
        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)

        s_val = [[0.2, 0.16, 0.16] for i in range(n_boxes)]
        t_val = [[0.25, 0.2, 0.2] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]

        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)
        # Get Random codes and bg rotation
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        out = []
        out_seg = []
        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, mode='val')
                out_i = seg2imgnet(out_seg_i, styles_code, is_sampling=True)
                #out_seg_i = tensor2label(out_seg_i, 8, imtype=np.float32, tile=True, forsave=True)
            out.append((out_i.cpu() + 1.0) / 2.0)
            # out_seg.append(torch.from_numpy(out_seg_i))
        out = torch.stack(out)
        out = out.squeeze(1)
        out_folder = join(img_out_path, 'random_rotation')
        makedirs(out_folder, exist_ok=True)
        for idx in range(out.shape[0]):
            img_grid = out[idx]
            save_image(img_grid, join(
                    out_folder, '%04d_%s.jpg' % (idx, 'random')))

    def render_object_rotation(self, img_out_path, batch_size=8, n_steps=32):

        gen = self.generator
        seg2imgnet = self.seg2imgnet
        bbox_generator = gen.bounding_box_generator
        n_boxes = bbox_generator.n_boxes
        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Get Random codes and bg rotation
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]

        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)

        out = []
        out_seg = []
        for step in range(n_steps):

            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)
            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, mode='val')
                ch = out_seg_i.shape[1]
                out_i = seg2imgnet(out_seg_i, styles_code, None, is_sampling=True)
                out_seg_i = tensor2label(out_seg_i, ch, imtype=np.float32, tile=True, forsave=True)
            out.append((out_i.cpu() + 1.0) / 2.0)
            out_seg.append(torch.from_numpy(out_seg_i))

        out = torch.stack(out)
        out_folder = join(img_out_path, 'rotation_object')
        out_seg = torch.stack(out_seg)
        out_seg_folder = join(img_out_path, 'rotation_object_seg')
        makedirs(out_folder, exist_ok=True)
        makedirs(out_seg_folder, exist_ok=True)

        self.save_video_and_images(
            out, out_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation))

        self.save_video_and_images(
            out_seg, out_seg_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation))

    def render_object_random(self, img_out_path, batch_size=1, n_steps=5000):

        gen = self.generator
        seg2imgnet = self.seg2imgnet

        bbox_generator = gen.bounding_box_generator
        n_boxes = bbox_generator.n_boxes
        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)

        s_val = [[0.2, 0.16, 0.16] for i in range(n_boxes)]
        t_val = [[0.25, 0.2, 0.2] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]

        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        out = []
        out_seg = []
        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)

            device = "cuda"
            styles_code = torch.randn(batch_size, 512, device=device)

            # Get Random codes and bg rotation
            latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, mode='val')
                out_i = seg2imgnet(out_seg_i, styles_code, None, is_sampling=True)
                #out_seg_i = tensor2label(out_seg_i, 8, imtype=np.float32, tile=True, forsave=True)
            out.append((out_i.cpu() + 1.0) / 2.0)
            # out_seg.append(torch.from_numpy(out_seg_i))
        out = torch.stack(out)
        out = out.squeeze(1)
        # out_folder = join(img_out_path, 'rotation_object')
        # out_seg = torch.stack(out_seg)
        # out_seg_folder = join(img_out_path, 'rotation_object_seg')
        # makedirs(out_folder, exist_ok=True)
        # makedirs(out_seg_folder, exist_ok=True)
        out_folder = join(img_out_path, 'random')
        makedirs(out_folder, exist_ok=True)
        for idx in range(out.shape[0]):
            img_grid = out[idx]
            save_image(img_grid, join(
                    out_folder, '%04d_%s.jpg' % (idx, 'random')))
        #
        # self.save_video_and_images(
        #     out_seg, out_seg_folder, name='rotation_object',
        #     is_full_rotation=is_full_rotation,
        #     add_reverse=(not is_full_rotation))

    def render_object_translation_horizontal(self, img_out_path, batch_size=16,
                                             n_steps=128, mode='t1'):
        gen = self.generator
        seg2imgnet = self.seg2imgnet
        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)
        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            if mode == 't1':
                if step < 10:
                    ti = t + [[x_val, i, 0.]]
                elif step < 20:
                    ti = t + [[i, 0, 0.]]
            elif mode == 't2':
                ti = t + [[i, 0, 0.]]
            else:
                ti = t + [[x_val, 0., i]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, mode='val')
                out_i = seg2imgnet(out_seg_i, styles_code, None, is_sampling=True)
                #out_seg_i = tensor2label(out_seg_i, 8, imtype=np.float32, tile=True, forsave=True)
            out.append((out_i.cpu() + 1.0) / 2.0)

        out = torch.stack(out)
        out_folder = join(img_out_path, 'translation_object_horizontal_{}'.format(mode))
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_horizontal',
            add_reverse=True)

    def render_object_translation_depth(self, img_out_path, batch_size=15,
                                        n_steps=32):
        gen = self.generator
        seg2imgnet = self.seg2imgnet

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            y_val = 0.5
        elif n_boxes == 2:
            t = [[0.4, 0.8, 0.]]
            y_val = 0.2

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[i, y_val, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, mode='val')
                out_i = seg2imgnet(out_seg_i, styles_code, None, is_sampling=True)
                # out_seg_i = tensor2label(out_seg_i, 8, imtype=np.float32, tile=True, forsave=True)
            out.append((out_i.cpu() + 1.0) / 2.0)

        out = torch.stack(out)
        out_folder = join(img_out_path, 'translation_object_depth')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_depth', add_reverse=True)

    def render_interpolation_random(self, img_out_path, batch_size=1, n_steps=5000, mode='app'):

        gen = self.generator
        seg2imgnet = self.seg2imgnet

        bbox_generator = gen.bounding_box_generator
        n_boxes = bbox_generator.n_boxes
        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)

        s_val = [[0.2, 0.16, 0.16] for i in range(n_boxes)]
        t_val = [[0.25, 0.2, 0.2] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]

        s, t, r = gen.get_transformations(s_val, t_val, r_val, batch_size)

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)
        app_code, pose_code = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        out = []
        for step in range(n_steps):

            if mode == 'app':
                device = "cuda"
                styles_code = torch.randn(batch_size, 512, device=device)
                # Get Random codes and bg rotation
                app_code_, _ = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
                latent_codes = [app_code_, pose_code]
            else:
                _, pose_code_ = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
                latent_codes = [app_code, pose_code_]

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, mode='val')
                out_i = seg2imgnet(out_seg_i, styles_code, is_sampling=True)
            out.append((out_i.cpu() + 1.0) / 2.0)

        out = torch.stack(out)
        out = out.squeeze(1)
        out_folder = join(img_out_path, 'random_{}'.format(mode))
        makedirs(out_folder, exist_ok=True)
        for idx in range(out.shape[0]):
            img_grid = out[idx]
            save_image(img_grid, join(
                    out_folder, '%04d_%s.jpg' % (idx, 'random')))

    def render_interpolation(self, img_out_path, batch_size=8, n_samples=2,
                             n_steps=32, mode='app'):

        gen = self.generator
        seg2imgnet = self.seg2imgnet
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_app_obj, z_pose_obj,  = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_obj.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        camera_matrices = gen.get_camera(batch_size=batch_size)
        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)

        out = []
        out_seg = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_ii, z_pose_obj]
                elif mode == 'pose':
                    latent_codes = [z_app_obj, z_ii]
                else:
                    latent_codes = None
                with torch.no_grad():
                    out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, mode='val')
                    out_i = seg2imgnet(out_seg_i, styles_code, None, is_sampling=True)
                    out_seg_i = tensor2label(out_seg_i, 8, imtype=np.float32, tile=True, forsave=True)
                out.append((out_i.cpu() + 1.0) / 2.0)
                out_seg.append(torch.from_numpy(out_seg_i))

        out = torch.stack(out)
        out_seg = torch.stack(out_seg)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True, img_n_steps=out.shape[0])

        out_folder = join(img_out_path, 'interpolate_%s_seg' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out_seg, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True, img_n_steps=out_seg.shape[0])

        # out = torch.stack(out)
        out = out.squeeze(1)
        out_folder = join(img_out_path, 'random_pose_{}'.format(mode))
        makedirs(out_folder, exist_ok=True)
        for idx in range(out.shape[0]):
            img_grid = out[idx]
            save_image(img_grid, join(
                    out_folder, '%04d_%s.jpg' % (idx, 'random')))

        # out = torch.stack(out_seg)
        out_seg = out_seg.squeeze(1)
        out_folder = join(img_out_path, 'random_pose_seg_{}'.format(mode))
        makedirs(out_folder, exist_ok=True)
        for idx in range(out_seg.shape[0]):
            img_grid = out_seg[idx]
            save_image(img_grid, join(
                    out_folder, '%04d_%s.jpg' % (idx, 'random')))

    def render_interpolation_as(self, img_out_path, batch_size=16, n_samples=2,
                             n_steps=32, mode='as'):

        gen = self.generator
        seg2imgnet = self.seg2imgnet

        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_app_obj, z_pose_obj,  = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_obj.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        camera_matrices = gen.get_camera(batch_size=batch_size)
        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        device = "cuda"

        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        out = []
        out_seg = []
        for j in range(n_samples):
            for step in range(n_steps):
                styles_code = torch.randn(batch_size, 512, device=device)
                with torch.no_grad():
                    out_seg_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, mode='val')
                    ch = out_seg_i.shape[1]
                    out_i = seg2imgnet(out_seg_i, styles_code, None, is_sampling=True)
                    out_seg_i = tensor2label(out_seg_i, ch, imtype=np.float32, tile=True, forsave=True)
                out.append((out_i.cpu() + 1.0) / 2.0)
                out_seg.append(torch.from_numpy(out_seg_i))

        out = torch.stack(out)
        out_seg = torch.stack(out_seg)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True, img_n_steps=out.shape[0])

        out_folder = join(img_out_path, 'interpolate_%s_seg' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out_seg, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True, img_n_steps=out_seg.shape[0])

    def render_camera_elevation(self, img_out_path, batch_size=64, n_steps=32):

        gen = self.generator
        seg2imgnet = self.seg2imgnet
        n_boxes = gen.bounding_box_generator.n_boxes
        r_range = [0.4167, 0.5]

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            [[0.5, 0.5, 0.5] for i in range(n_boxes)],
            [0.5 for i in range(n_boxes)],
            batch_size,
        )

        device = "cuda"
        styles_code = torch.randn(batch_size, 512, device=device)

        out = []
        out_seg = []
        for step in range(n_steps):
            v = step * 1.0 / (n_steps - 1)
            r = r_range[0] + v * (r_range[1] - r_range[0])
            camera_matrices = gen.get_camera(val_v=r, batch_size=batch_size)
            with torch.no_grad():
                out_seg_i = gen(batch_size, latent_codes, camera_matrices, transformations, mode='val')
                out_i = seg2imgnet(out_seg_i, styles_code, is_sampling=True)
                ch = out_seg_i.shape[1]
                out_seg_i = tensor2label(out_seg_i, ch, imtype=np.float32, tile=True, forsave=True)
                out_seg.append(torch.from_numpy(out_seg_i))
                out.append((out_i.cpu() + 1.0) / 2.0)

        out = torch.stack(out)
        out_seg = torch.stack(out_seg)

        out_folder = join(img_out_path, 'camera_elevation')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate',
            is_full_rotation=True, img_n_steps=out.shape[0])

        out_folder = join(img_out_path, 'camera_elevation_seg')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out_seg, out_folder, name='interpolate',
            is_full_rotation=True, img_n_steps=out_seg.shape[0])

    ##################
    # Helper functions
    def write_video(self, out_file, img_list, n_row=5, add_reverse=False,
                    write_small_vis=True):
        n_steps, batch_size = img_list.shape[:2]
        nrow = n_row if (n_row is not None) else int(sqrt(batch_size))
        img = [(255*make_grid(img, nrow=nrow, pad_value=1.).permute(
            1, 2, 0)).cpu().numpy().astype(np.uint8) for img in img_list]
        if add_reverse:
            img += list(reversed(img))
        if write_small_vis:
            img = [(255*make_grid(img, nrow=batch_size, pad_value=1.).permute(
                1, 2, 0)).cpu().numpy().astype(
                    np.uint8) for img in img_list[:, :9]]
            if add_reverse:
                img += list(reversed(img))
            imageio.mimwrite(
                (out_file[:-4] + '_sm.mp4'), img, fps=30, quality=4)

    def save_video_and_images(self, imgs, out_folder, name='rotation_object',
                              is_full_rotation=False, img_n_steps=6,
                              add_reverse=False):
        # Save video
        out_file_video = join(out_folder, '%s.mp4' % name)
        self.write_video(out_file_video, imgs, add_reverse=add_reverse)
        #Save images
        n_steps, batch_size = imgs.shape[:2]
        if is_full_rotation:
            idx_paper = np.linspace(
                0, n_steps - n_steps // img_n_steps, n_steps
            ).astype(np.int)
        else:
            idx_paper = np.linspace(0, n_steps - 1, n_steps).astype(np.int)
        for idx in range(batch_size):
            img_grid = imgs[idx_paper, idx]
            for j in range(img_grid.shape[0]):
                save_image(img_grid[j], join(
                        out_folder, '%04d_%04d_%s.jpg' % (idx, j, name)))