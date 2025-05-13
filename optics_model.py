import opticsmodel as om
import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
import torchvision.transforms.functional as Ft
from tqdm import tqdm


class OpticalNet(nn.Module):
    def __init__(self, config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(OpticalNet, self).__init__()
        with open(config) as f:
            self.args = yaml.load(f, Loader=yaml.FullLoader)
        torch.set_default_dtype(torch.float64)
        torch.set_default_device(device)
        self.lens = om.System(self.args['LENS_PATH'], ref_wav=self.args['REF_WAV'])
        self.analysis = om.Analysis(self.lens, torch.tensor(self.args['views']), torch.tensor(self.args['wavelengths']))

    def update(self):
        self.lens._update_transforms()
        self.lens._pre_calculation()

    def get_optimizer_Adam(self):
        params_list, params_name = self.lens.get_variables()
        lr_base = self.args['lr_base']
        efl = self.lens.Effective_Focal_Length
        params_group = []
        for name, params in zip(params_name, params_list):
            if name.endswith('c'):
                params_group.append({'params': params, 'lr': lr_base/efl})
            elif name.endswith('d'):
                params_group.append({'params': params, 'lr': lr_base*efl})
            elif name.endswith('k'):
                params_group.append({'params': params, 'lr': lr_base})
            elif name.endswith('ai4'):
                params_group.append({'params': params, 'lr': lr_base/efl**3})
            elif name.endswith('ai6'):
                params_group.append({'params': params, 'lr': lr_base/efl**5})
            elif name.endswith('ai8'):
                params_group.append({'params': params, 'lr': lr_base/efl**7})
            elif name.endswith('ai10'):
                params_group.append({'params': params, 'lr': lr_base/efl**9})
            elif name.endswith('ai12'):
                params_group.append({'params': params, 'lr': lr_base / efl ** 11})
            elif name.endswith('ai14'):
                params_group.append({'params': params, 'lr': lr_base / efl ** 13})
            elif name.endswith('ai16'):
                params_group.append({'params': params, 'lr': lr_base / efl ** 15})

        return torch.optim.Adam(params_group)


    def loss_fnumber(self, stacks_list, target, weight):
        f_number = self.lens.F_Number
        f_number_loss = torch.nn.functional.relu(f_number - target) * weight
        return f_number_loss, f_number

    def loss_ray_path(self, stacks_list, target, weight):
        # depreciated
        ray_path_loss = 0
        ray_path_eval = 0
        for stacks in stacks_list[1:]:
            valid_map = stacks['ray'].valid_map
            ray_path_surfaces = stacks['t']
            for ray_path in ray_path_surfaces[1:]:
                ray_path_eval += torch.nn.functional.relu(target-ray_path[valid_map]).mean()
                ray_path_loss += torch.nn.functional.relu(target-ray_path[valid_map]).mean()*weight
        return ray_path_loss, ray_path_eval

    def loss_thickness(self, stacks_list, target, weight):
        d = self.analysis.eval_surf_distance()
        thickness_eval = torch.min(d)
        thickness_loss = torch.nn.functional.relu(target-d).mean()*weight*d.shape[0]
        return thickness_loss, thickness_eval

    def loss_effl(self, stacks_list, target, weight):
        effl = self.lens.Effective_Focal_Length
        effl_loss = torch.nn.functional.smooth_l1_loss(effl, torch.tensor(target), beta=0.2) * weight
        return effl_loss, effl

    def loss_ttl(self, stacks_list, target, weight):
        ttl = self.lens.TTL
        ttl_loss = torch.nn.functional.relu((ttl - target)*ttl)*weight
        return ttl_loss, ttl

    def loss_spots(self, stacks_list, target, weight):
        # 相对于主光线的点列rms误差
        ref_ray = stacks_list[0]['ray']
        image_point = torch.mean(ref_ray.o[ref_ray.valid_map], dim=0)
        spots_loss = 0
        spots_eval = 0
        wav_num = len(stacks_list)-1
        for stacks in stacks_list[1:]:
            valid_map = stacks['ray'].valid_map
            ray_points = stacks['ray'].o[valid_map]
            rms = torch.sqrt(((ray_points-image_point)**2).sum(-1).mean())
            spots_eval += rms
            spots_loss += torch.relu(rms-target)*weight
        spots_eval /= wav_num
        spots_loss /= wav_num
        return spots_loss, spots_eval

    def loss_distortion(self, stacks_list, target, weight):
        ref_ray = stacks_list[0]['ray']
        image_point = torch.mean(ref_ray.o[ref_ray.valid_map], dim=0)
        image_point_r = torch.sqrt((image_point[:2]**2).sum())
        ideal_image_point_r = self.lens.Distortion_Focal_Length * torch.tan(self.lens[0].angle_fov).abs()
        distortion = (image_point_r-ideal_image_point_r)/ideal_image_point_r
        distortion_loss = torch.nn.functional.relu(distortion.abs() - target) * weight
        return distortion_loss, distortion.abs()

    def get_optical_loss_dict(self, stacks_list):
        """
        根据args中的loss项，对stacks进行分析
        """
        optical_loss_dict = dict()
        eval_dict = dict()
        for name in self.args['loss_functions'].keys():
            l_fn = getattr(self, "loss_%s" % name)
            optical_loss_dict[name], eval_dict[name] = l_fn(stacks_list, *self.args['loss_functions'][name])
        return optical_loss_dict, eval_dict

    @property
    def field_delta(self):
        image_resolution = self.args['image_resolution']
        field_delta = 2 / (image_resolution - 1)
        return field_delta

    def get_kernel_field(self, fov, azimuth):
        """
        从一个随机采样的视场下，生成当前模型的psf
        :param fov:
        :return:
        """
        kernel_size = self.args['kernel_size']
        image_delta = self.args['image_delta']

        stack_list = []
        ref_ray = self.analysis.sample_ray_fov_azimuth(self.analysis.system.ref_wav, fov, azimuth, self.args['pupil_sampling'])
        stack = self.analysis.system.propagate(ref_ray, clip=True)
        stack_list.append(stack)
        # ref_ray = stack['ray']
        # image_point = torch.mean(ref_ray.o[ref_ray.valid_map], dim=0)
        chief_ray_o = torch.mean(ref_ray.o[stack['ray'].valid_map], dim=0)
        chief_ray_o[0].data *= 0
        chief_ray_d = ref_ray.d[0]
        chife_ray = om.Ray(chief_ray_o.unsqueeze(0), chief_ray_d.unsqueeze(0), ref_ray.wavelength)
        image_point = self.analysis.system.propagate(chife_ray, clip=True)['ray'].o[0]
        line_sample = torch.linspace(-1, 1, kernel_size) / 2 * image_delta * (kernel_size - 1)
        x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        psf_coor = image_point + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
        image_resolution = self.args['image_resolution']
        r = (image_resolution-1) * image_delta / 2
        center_filed_x, center_filed_y = image_point[0] / r, image_point[1] / r
        field_center = torch.stack([center_filed_x, center_filed_y])
        psf_wavs = []
        for wavelength in self.analysis.wavelengths:
            wave_num = 2 * torch.pi / wavelength
            ray = self.analysis.sample_ray_fov_azimuth(wavelength=wavelength, fov=fov, azimuth=azimuth, pupil_sampling=self.args['pupil_sampling'])
            stacks = self.analysis.system.propagate(ray, clip=True)
            stack_list.append(stacks)
            ray = stacks['ray']
            psf = torch.flipud(om.CoherentPsfOp.apply(ray.o[ray.valid_map],ray.d[ray.valid_map],psf_coor,ray.opd[ray.valid_map],wave_num).T)
            psf = psf / psf.sum()
            psf_wavs.append(psf)
        return torch.stack(psf_wavs), field_center, stack_list


    def rendering_patch(self, fov, azimuth, img):
        """
        用于e2e渲染数据的
        1.根据optics_feature计算psf以及视场
        2.渲染退化图像用于后续的网络输入，此时的 RGB通道 附加上xy视场信息[fov,azimuth]相对于最大视场
        可以分成两个函数去写，先获取psf与视场信息；随后插值，并对数据集中图片进行处理
        ！！ x、y fov delta 根据image——delta换算
        """
        kernels, field_center, stack_list = self.get_kernel_field(fov, azimuth)
        chromatic_kernel = torch.zeros([3, self.args['kernel_size'], self.args['kernel_size']])
        for i, psf in enumerate(kernels):
            chromatic_kernel[0] += psf * self.args['b'][i]
            chromatic_kernel[1] += psf * self.args['g'][i]
            chromatic_kernel[2] += psf * self.args['r'][i]
        for channel in range(3):
            chromatic_kernel[channel] /= chromatic_kernel[channel].sum()
        blur_img = F.conv2d(img, torch.flip(chromatic_kernel.unsqueeze(1).to(torch.float32), dims=[-1, -2]), groups=3)
        img_size = blur_img.shape[-1]
        line_sample = torch.linspace(-1, 1, img_size) / 2 * (img_size - 1) * self.field_delta
        field_x, field_y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        field = (torch.stack([field_x, field_y], dim=-1) + field_center).to(torch.float32)
        return blur_img, field.permute(2, 0, 1).repeat(blur_img.shape[0], 1, 1, 1), stack_list

    @torch.no_grad()
    def rendering_psf_map(self, H, W, patch_itv, file_name='psf_map.png', save=False):
        """
        用于生成psf map 可视化卷积,也保存kernel：Hn,Wn,CHW
        """
        H_nums = int(H / patch_itv)
        W_nums = int(W / patch_itv)
        psf_map = torch.zeros(3, H_nums*self.args['kernel_size'], W_nums*self.args['kernel_size'])
        dfl = self.lens.dfl
        total_count = H_nums * W_nums
        kernel_set = torch.zeros(H_nums, W_nums, 3, self.args['kernel_size'], self.args['kernel_size'])
        field_set = torch.zeros(H_nums, W_nums, patch_itv, patch_itv, 2)
        with tqdm(total=total_count) as pbar:
            pbar.set_description("Rendering psf map")
            for h_index in range(H_nums):
                for w_index in range(W_nums):
                    physical_center_y = torch.tensor((-H // 2 + (h_index + 1 / 2) * patch_itv) * self.args['image_delta'])
                    physical_center_x = torch.tensor(((w_index + 1 / 2) * patch_itv - W // 2) * self.args['image_delta'])
                    r = torch.sqrt(physical_center_x**2+physical_center_y**2)
                    fov = torch.arctan2(r, dfl).rad2deg()
                    azimuth = torch.arctan2(physical_center_x, physical_center_y).rad2deg()

                    # 一步畸变矫正
                    ref_ray = self.analysis.sample_ray_fov_azimuth(self.analysis.system.ref_wav, fov, azimuth, self.args['pupil_sampling'])
                    stack = self.analysis.system.propagate(ref_ray, clip=True)
                    ref_ray = stack['ray']
                    image_point = torch.mean(ref_ray.o[ref_ray.valid_map], dim=0)
                    dist_first = (torch.sqrt((image_point[:2] ** 2).sum()) - r) / r
                    fov = torch.arctan(torch.tan(fov.deg2rad()) / (1 + dist_first)).rad2deg()

                    kernel, field_center, _ = self.get_kernel_field(fov, azimuth)
                    line_sample = torch.linspace(-1, 1, patch_itv) / 2 * (patch_itv - 1) * self.field_delta
                    field_x, field_y = torch.meshgrid(line_sample, line_sample, indexing='ij')
                    field = (torch.stack([field_x, field_y], dim=-1) + field_center).to(torch.float32)
                    field_set[h_index, w_index] = field
                    kernel_set[h_index, w_index] = kernel
                    for i in range(3):
                        kernel[i] /= kernel[i].max()
                    itv = self.args['kernel_size']
                    psf_map[:, h_index*itv:(h_index+1)*itv, w_index*itv:(w_index+1)*itv] = torch.flipud(kernel)
                    pbar.update(1)
            pbar.close()
        if save:
            Ft.to_pil_image(psf_map.cpu()).save(file_name)
        return kernel_set, field_set

    @torch.no_grad()
    def rendering_psf_line(self, nums, file_name='psf_line.pdf', save=False):
        """
        用于生成psf map 可视化卷积,也保存kernel：Hn,Wn,CHW
        """
        psf_map = torch.zeros(3, self.args['kernel_size'], nums * self.args['kernel_size'])
        views = torch.linspace(0,self.args['max_view'], nums)
        azimuth = torch.tensor(0.)
        with tqdm(total=nums) as pbar:
            pbar.set_description("Rendering psf line")
            for idx, v in enumerate(views):
                kernel, _, _ = self.get_kernel_field(v, azimuth)
                for i in range(3):
                    kernel[i] /= kernel[i].max()
                itv = self.args['kernel_size']
                psf_map[:, :, idx * itv:(idx + 1) * itv] = torch.flipud(kernel)
                pbar.update(1)
            pbar.close()
        if save:
            Ft.to_pil_image(psf_map.cpu()).save(file_name)
        return psf_map

    @torch.no_grad()
    def rendering_image(self, img: torch.Tensor, kernel_set: torch.Tensor, patch_size=250, patch_itv=200):
        """
        :param img: BCHW
        :param kernel_set:  Hn Wn CHW
        :param patch_itv:  blured patch size
        :param patch_size: patch for convolution
        :return: blured img + fov B5HW
        """
        H, W = img.shape[-2:]
        H_nums, W_nums = kernel_set.shape[:2]
        field_delta = 2 / (self.args['image_resolution'] - 1)
        H_sample = torch.linspace(-1, 1, H) / 2 * (H - 1) * field_delta
        W_sample = torch.linspace(-1, 1, W) / 2 * (W - 1) * field_delta
        field_x, field_y = torch.meshgrid(H_sample, W_sample, indexing='ij')
        field = (torch.stack([field_x, field_y], dim=-1)).to(torch.float32)
        img_degraded = torch.zeros_like(img)
        pad_length = int((patch_size - patch_itv) / 2)
        img = torch.nn.functional.pad(img, (pad_length, pad_length, pad_length, pad_length))
        for h_index in range(H_nums):
            for w_index in range(W_nums):
                # crop the patch for convolution
                patch = img[:, :, h_index * patch_itv: h_index * patch_itv + patch_size, w_index * patch_itv: w_index * patch_itv + patch_size]
                # 接下来计算这个patch中心的kernel
                kernel = kernel_set[h_index, w_index]
                patch_degraded = F.conv2d(patch, torch.flip(kernel.unsqueeze(1).to(torch.float32), dims=[-1, -2]), groups=3)
                img_degraded[:, :, h_index * patch_itv: h_index * patch_itv + patch_itv, w_index * patch_itv: w_index * patch_itv + patch_itv] = \
                    patch_degraded[:, :, pad_length: pad_length + patch_itv, pad_length: pad_length + patch_itv]
        blur_img = torch.cat([img_degraded, field.permute(2, 0, 1).repeat(img_degraded.shape[0], 1, 1, 1)], dim=1)
        return blur_img

    @torch.no_grad()
    def crop_patch(self, img, patch_size=500, kernel_size=31):
        pad_size = (kernel_size // 2) * 2
        [B, C, H_img, W_img] = img.shape
        H_num = int((H_img - pad_size) / patch_size)
        W_num = int((W_img - pad_size) / patch_size)
        field_delta = 2 / (self.args['image_resolution'] - 1)
        H_sample = torch.linspace(-1, 1, H_img) / 2 * (H_img - 1) * field_delta
        W_sample = torch.linspace(-1, 1, W_img) / 2 * (W_img - 1) * field_delta
        # field_x, field_y = torch.meshgrid(H_sample, W_sample, indexing='ij')
        field_y, field_x = torch.meshgrid(H_sample, W_sample, indexing='ij')
        field = (torch.stack([field_x, field_y], dim=-1)).to(torch.float32)
        patch_list_hw = []
        field_list_hw = []
        for h_index in range(H_num):
            patch_list_w=[]
            field_list_w=[]
            for w_index in range(W_num):
                patch = img[:, :, patch_size * h_index: patch_size * (h_index + 1) + pad_size + 100,
                            patch_size * w_index: patch_size * (w_index + 1) + pad_size + 100]
                field_patch = field[pad_size//2 + patch_size * h_index: patch_size * (h_index + 1) + pad_size//2 + 100,
                                    pad_size//2 + patch_size * w_index: patch_size * (w_index + 1) + pad_size//2 + 100, :]
                field_list_w.append(field_patch)
                patch_list_w.append(patch)
            field_list_hw.append(field_list_w)
            patch_list_hw.append(patch_list_w)
        return patch_list_hw, field_list_hw

    @torch.no_grad()
    def sew_up_img(self, out_patch_list, patch_size=500, kernel_size=31, img_size=[3000, 4000]):
        rgb = torch.zeros((out_patch_list[0].shape[0], out_patch_list[0].shape[1], img_size[0], img_size[1]))
        pad_size = kernel_size // 2
        for patch_index in range(len(out_patch_list)):
            # w seq first, h seq second
            h_index = patch_index // int(img_size[1] / patch_size)
            w_index = patch_index - h_index * int(img_size[1] / patch_size)
            patch_data = out_patch_list[patch_index].clone()
            rgb[:, :, h_index * patch_size: (h_index + 1) * patch_size, w_index * patch_size: (w_index + 1) * patch_size] = patch_data
        return rgb

    @torch.no_grad()
    def blur_patch(self, img, kernel):
        blur_img = F.conv2d(img, torch.flip(kernel.unsqueeze(1), dims=[-1, -2]), groups=3)
        return blur_img

