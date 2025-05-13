

import matplotlib.pyplot as plt
import opticsmodel
from .surfaces import *
from .basics import Ray, DEFAULT_WAVELENGTH


class Analysis(nn.Module):
    def __init__(self, system: opticsmodel.System, views, wavelengths, vignetting=False):
        super().__init__()
        self.system = system
        self.views = views
        self.wavelengths = wavelengths
        if vignetting:
            self.set_vignetting()
        else:
            self.vig = {k.item(): {'VDX': 0., 'VDY': 0., 'VCX': 0., 'VCY': 0.} for k in self.views}
        self.update_system()

    # ====================================================================================
    # System Viewers
    # ====================================================================================
    def plot_setup_2d(self, ax=None, fig=None, show=True, color='k'):
        """
        Plot elements in 2D.
        """
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            show = False

        # to world coordinate
        def plot(ax, z, x, color):
            ax.plot(z.cpu().detach().numpy(), x.cpu().detach().numpy(), color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.origin[2].cpu().item()
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R
            APERTURE_WEDGE_HEIGHT = 0.15 * R

            # wedge length
            z = torch.linspace(d - APERTURE_WEDGE_LENGTH, d + APERTURE_WEDGE_LENGTH, N)
            x = -R * torch.ones(N)
            plot(ax, z, x, color)
            x = R * torch.ones(N)
            plot(ax, z, x, color)

            # wedge height
            z = d * torch.ones(N)
            x = torch.linspace(R, R + APERTURE_WEDGE_HEIGHT, N)
            plot(ax, z, x, color)
            x = torch.linspace(-R - APERTURE_WEDGE_HEIGHT, -R, N)
            plot(ax, z, x, color)

        # if there is only one surface, then it has to be the aperture
        # draw surface
        for i, s in enumerate(self.system):
            if i == 0:
                continue
            if i == self.system.aperture_ind:
                draw_aperture(ax, self.system[self.system.aperture_ind], color)
                continue
            # draw surface
            r = torch.linspace(-s.r, s.r, 7 * s.SURFACE_SAMPLING)
            z = s.surface_with_offset(r, torch.zeros(len(r)))
            plot(ax, z, r, color)
            # draw boundary
            if s.Material.nd > 1.0003:
                s_post = self.system.surfaces[i + 1]
                r_post = s_post.r
                r = s.r

                sag = s.surface_with_offset(r, 0.0).squeeze()
                sag_post = s_post.surface_with_offset(r_post, 0.0).squeeze()

                z = torch.stack((sag, sag_post))
                x = torch.stack([r, r_post])

                plot(ax, z, x, color)
                plot(ax, z, -x, color)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        plt.title("".join([self.system.LensName, " Layout 2D"]))
        if show:
            plt.show()
        return ax, fig

    def plot_setup_2d_with_trace(self, num_rays=7, show=True):
        """
        Plot elements and rays in different views.
        M: number of rays in yz plane
        R: radius of rays bundle
        """
        wavelength = self.system.ref_wav

        colors_list = 'bgrymck'
        ax, fig = self.plot_setup_2d(show=False)

        # plot rays
        for i, view in enumerate(self.views):
            ray = self.sample_ray_yz(wavelength, view=view, num_rays=num_rays)
            stacks = self.system.propagate(ray)
            oss = torch.stack(stacks['o']).permute(1, 0, 2)
            valid = torch.stack(stacks['valid']).permute(1, 0)
            for os, v in zip(oss, valid):
                y = os[v, 1].cpu().detach().numpy()
                z = os[v, 2].cpu().detach().numpy()
                ax.plot(z, y, colors_list[i], linewidth=1.0)

        if show:
            plt.show()
        return ax, fig

    # =================================
    # Rays and Spots
    # =================================
    def sample_ray_yz(self, wavelength, view=torch.tensor(0.), num_rays=7):
        """
        sample rays in yz plane, use to plot ray trace
        """
        ref_rays = self.ref_rays[view.item(), wavelength.item()]
        y_up = ref_rays.o[1, 1]
        y_down = ref_rays.o[2, 1]
        y = torch.linspace(y_down, y_up, num_rays)

        o = torch.stack((torch.zeros_like(y), y, torch.ones_like(y) * self.system.entrance_pupil_position), dim=-1)
        self.system[0].update(view)
        if self.system[0].finite:
            d = self.system[0].map(origin_sample=o)
        elif self.system[0].infinite:
            d = self.system[0].map().unsqueeze(0).repeat(num_rays, 1)
        return Ray(o, d, wavelength)

    @torch.no_grad()
    def calc_vignetting(self, view=torch.tensor(0.), clip=True):
        """
        prepares valid rays that successfully propagate through the system
        """
        # update the field-of-view of system
        self.system[0].update(view)
        # maximum radius input
        R = self.system.entrance_pupil_radius * 1.1
        wavelength = self.system.ref_wav
        APERTURE_SAMPLING = 201

        x_e, y_e = torch.meshgrid(torch.linspace(-R, R, APERTURE_SAMPLING),
                                  torch.linspace(-R, R, APERTURE_SAMPLING), indexing='ij')
        x_e, y_e = x_e.reshape(-1), y_e.reshape(-1)
        # generate rays and find valid map
        o = torch.stack((x_e, y_e, torch.ones_like(x_e) * self.system.entrance_pupil_position), dim=-1)
        if self.system[0].finite:
            d = self.system[0].map(origin_sample=o)
        elif self.system[0].infinite:
            d = self.system[0].map().unsqueeze(0).unsqueeze(0).repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1)
        ray = Ray(o.reshape(-1, 3), d.reshape(-1, 3), wavelength=wavelength)
        # make ray clipped by surf.r
        ray_stop = self.system.propagate(ray, stop=self.system.aperture_ind + 1, clip=clip)['ray']
        ray_image = self.system.propagate(ray_stop, start=self.system.aperture_ind + 1, clip=clip)['ray']
        # find bounding box
        xe, ye = x_e[ray_image.valid_map], y_e[ray_image.valid_map]

        # roughly estimate
        VDX, VDY = xe.mean(), ye.mean()
        # actually we should search on S T direction
        VCX = 1 - (xe.max() - xe.min()) / (2 * self.system.entrance_pupil_radius)
        VCY = 1 - (ye.max() - ye.min()) / (2 * self.system.entrance_pupil_radius)

        """
        xs, ys = ray_stop.o[ray_image.valid_map, 0], ray_stop.o[ray_image.valid_map, 1]
        # -----------search with fine mesh----------------
        def fine_search(x,y):
            x_e, y_e = torch.meshgrid(torch.linspace(x - R * 0.01, x + R * 0.01, APERTURE_SAMPLING),
                                      torch.linspace(y - R * 0.01, y + R * 0.01, APERTURE_SAMPLING), indexing='ij')
            o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=-1)
            ray = Ray(o.reshape(-1, 3), d.reshape(-1, 3), wavelength=wavelength)
            # make ray clipped by surf.r
            ray_stop = self.system.propagate(ray, stop=self.system.aperture_ind + 1, clip=clip)['ray']
            ray_image = self.system.propagate(ray_stop, start=self.system.aperture_ind + 1, clip=clip)['ray']
            return ray_stop.o[ray_image.valid_map]
        # if we got two symmetry points just choose the first
        ys_max = fine_search(xe[ys == ys.max()][0], ye[ys == ys.max()][0])[..., 1].max()
        ys_min = fine_search(xe[ys == ys.min()][0], ye[ys == ys.min()][0])[..., 1].min()
        xs_max = fine_search(xe[xs == xs.max()][0], ye[xs == xs.max()][0])[..., 0].max()
        xs_min = fine_search(xe[xs == xs.min()][0], ye[xs == xs.min()][0])[..., 0].min()

        # set vignetting according to rays valid on stop
        VDX = (xs_max + xs_min) / (2 * self.system[self.system.aperture_ind].r)
        VCX = 1 - (xs_max - xs_min) / (2 * self.system[self.system.aperture_ind].r)
        VDY = (ys_max + ys_min) / (2 * self.system[self.system.aperture_ind].r)
        VCY = 1 - (ys_max - ys_min) / (2 * self.system[self.system.aperture_ind].r)
        """

        return {'VDX': VDX, 'VDY': VDY, 'VCX': VCX, 'VCY': VCY}

    @torch.no_grad()
    def sample_ray_vig(self, wavelength=torch.tensor(DEFAULT_WAVELENGTH), view=torch.tensor(0.), pupil_sampling=31, sampling='grid'):
        """
        sample rays from view with different sampling: grid,radial
        rays fulfill the entrance pupil
        """
        # update the field-of-view of system
        self.system[0].update(view)
        ref_rays = self.ref_rays[view.item(), wavelength.item()]
        ymax, ymin, xmax, xmin = ref_rays.o[1, 1], ref_rays.o[2, 1], ref_rays.o[3, 0], ref_rays.o[4, 0]
        """
        # uniform sampling
        delta_x = (xmax - xmin) / (pupil_sampling - 1)
        delta_y = delta_x * torch.cos(view.deg2rad())
        pupil_sampling_y = int((ymax - ymin) / delta_y) + 1
        if sampling == 'grid':
            x, y = torch.meshgrid(torch.linspace(xmin, xmax, pupil_sampling), torch.linspace(ymin, ymax, pupil_sampling_y), indexing='ij')
        else:
            raise ValueError(f'not implement pupil sampling method {sampling}')
        """
        x, y = torch.meshgrid(torch.linspace(xmin, xmax, pupil_sampling), torch.linspace(ymin, ymax, pupil_sampling), indexing='ij')
        x, y = x.reshape(-1), y.reshape(-1)
        o = torch.stack((x, y, torch.ones_like(x) * self.system.entrance_pupil_position), dim=-1)
        d = self.system[0].map().repeat(o.shape[0], 1)
        rays = Ray(o, d, wavelength)
        # optical path calculation with propagate
        chief_ray_o = ref_rays.o[0, :]
        rays.opd += torch.einsum('ij,ij->i', rays.o - chief_ray_o, rays.d)
        return rays

    @torch.no_grad()
    def sample_ray_fov_azimuth(self, wavelength=torch.tensor(DEFAULT_WAVELENGTH), fov=torch.tensor(0.), azimuth=torch.tensor(0.), pupil_sampling=31):
        """
        sample rays from view with different sampling: grid,radial
        rays fulfill the entrance pupil
        """
        # update the field-of-view of system
        self.system[0].update(fov, azimuth)
        r = self.system.entrance_pupil_radius * 1.1
        x, y = torch.meshgrid(torch.linspace(-r, r, pupil_sampling), torch.linspace(-r, r, pupil_sampling), indexing='ij')
        x, y = x.reshape(-1), y.reshape(-1)
        o = torch.stack((x, y, torch.ones_like(x) * self.system.entrance_pupil_position), dim=-1)
        d = self.system[0].map().repeat(o.shape[0], 1)
        rays = Ray(o, d, wavelength)
        # optical path calculation with propagate
        rays.opd += torch.einsum('ij,ij->i', rays.o - rays.o[o.shape[0]//2], rays.d)
        return rays

    def spot_diagram(self, views=None, wavelengths=None, pupil_sampling=100):
        """
        plot spot diagram
        return mean wavelengths spots for different views
        """
        if views is None:
            views = self.views
        if wavelengths is None:
            wavelengths = self.wavelengths

        lim = 0
        ps_dic = {}
        image_point = []
        for i, view in enumerate(views):
            ref_image_point = self.image_point(view)[:2]
            image_point.append(ref_image_point[1].cpu().detach().numpy())
            for j, wavelength in enumerate(wavelengths):
                ray = self.sample_ray_vig(wavelength=wavelength, view=view, pupil_sampling=pupil_sampling)
                ray = self.system.propagate(ray, clip=True)['ray']
                ps = (ray.o[ray.valid_map, :2] - ref_image_point).cpu().detach().numpy()
                ps_dic[i, j] = ps
                if np.abs(ps).max() > lim:
                    lim = np.abs(ps).max()

        colors_list = 'bgrymck'
        for i, view in enumerate(views):
            fig = plt.figure()
            ax = plt.axes()
            ps_view = []
            for j, wavelength in enumerate(wavelengths):
                ps_view.extend(ps_dic[i, j])
                x = ps_dic[i, j][..., 0]
                y = ps_dic[i, j][..., 1]
                ax.scatter(x, y, color=colors_list[j], s=0.1, label=f'{wavelength.item() * 1e6} nm')
            rms = np.sqrt(np.mean(np.sum(np.asarray(ps_view) ** 2, axis=-1)))
            plt.gca().set_aspect('equal', adjustable='box')
            xlims = [-2 * lim, 2 * lim]
            ylims = [-2 * lim, 2 * lim]
            plt.xlim(*xlims)
            plt.ylim(*ylims)
            ax.set_aspect(1. / ax.get_data_ratio())
            ax.legend()
            plt.xlabel('x [mm]')
            plt.ylabel('y [mm]')

            ax.set_title(f'view: {view} degree, ima: {image_point[i]:.3f} mm, RMS: {rms:.4f}')
        plt.show()

    def psf_coherent(self, pupil_sampling, image_sampling, image_delta, wavelength=None, view=None):
        """
        coherent superposition with complex amplitude!
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths) // 2]
        if view is None:
            view = self.views[-1]
        # rays sampling
        ray = self.sample_ray_vig(wavelength=wavelength, view=view, pupil_sampling=pupil_sampling)
        ray = self.system.propagate(ray)['ray']
        # initialize the sample complex amplitude mat on image plane
        # with torch.no_grad():
        #     image_point = self.get_real_image_point(view)
        image_point = self.image_point(view)
        line_sample = torch.linspace(-1, 1, image_sampling) / 2 * image_delta * (image_sampling - 1)
        x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        psf_coor = image_point + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
        wave_num = torch.tensor(2 * np.pi / wavelength.item())
        # calculate the complex amplitude of each sampled image position
        r = psf_coor[:, :, None, :] - ray.o[None, None, ray.valid_map, :]
        # inner production for the final length, dim:x,y,ray,data
        dr = torch.einsum('ijkl,ijkl->ijk', ray.d[None, None, ray.valid_map, :], r)
        # complex amplitude
        amp = torch.einsum('ijk->ij', torch.exp(((ray.opd[None, None, ray.valid_map] + dr) * wave_num) * (0 + 1j)))
        psf = torch.flipud(torch.abs(amp).T ** 2)
        psf = psf / psf.sum()
        return psf

    def single_ray_trace(self, Px=0, Py=0, wavelength=None, view=None):
        """
        Tracing a single ray with
        the normalized pupil coordinates [Px, Py],
        the [wavelength],
        the [view] (in degree),
        and use the [global coordinates] or [local coordinates]
        """
        if wavelength is None:
            wavelength = self.system.ref_wav
        if view is None:
            view = self.views[-1]
        # update the field of view of this system
        self.system[0].update(view)
        ray = self.system.aim_ray(Px, Py, view=view, wavelength=wavelength)
        # propagate the ray and get its intersection on each surface
        stacks = self.system.propagate(ray, start=1, stop=None)
        ###############################
        # print the trace of single ray
        ###############################
        # the head
        print('Ray Trace Data \n')
        print('Lens Title: {}'.format(self.system.LensName))

        print('Units         :   Millimeters')
        print('Wavelength    :   {:0.6f}  um'.format(wavelength * 1e3))
        print('Coordinates   :   Global coordinates relative to surface 1 \n')

        print('Field-of-View :   {:f} degree (represent in angle) '.format(view))
        print('Normalized X Pupil Coord (Px) :      {:0.10f}'.format(Px))
        print('Normalized Y Pupil Coord (Py) :      {:0.10f} \n'.format(Py))

        # rays data
        print('Real Ray Trace Data: \n')
        itv = 4  # interval between rows
        print('Surf' + ' ' * itv + ' ' * 4 + 'X-coordinate' +
              ' ' * itv + ' ' * 5 + 'Y-coordinate' +
              ' ' * itv + ' ' * 5 + 'Z-coordinate' +
              ' ' * itv + ' ' * 5 + 'X-cosine' +
              ' ' * itv + ' ' * 5 + 'Y-cosine' +
              ' ' * itv + ' ' * 5 + 'Z-cosine' +
              ' ' * itv + ' ' * 5 + 'Angle' +
              ' ' * itv + ' ' * 5 + 'Path length')
        oss, dss, tss, iss = stacks['o'], stacks['d'], stacks['t'], stacks['in']
        for idx in range(len(oss)):  # for oss and dss the 1 dimension is the surface index
            os, ds, ts, ai = oss[idx].squeeze(0), dss[idx].squeeze(0), tss[idx], iss[idx]
            msg = 'OBJ' if idx == 0 else '{:>3d}'.format(idx)
            if idx == 0:
                if self.system[0]._type == 'infinite':
                    msg += ' ' * itv + ' ' * 9 + 'Infinity' + ' ' * itv + ' ' * 9 + 'Infinity' + \
                           ' ' * itv + ' ' * 9 + 'Infinity'
                elif self.system[0]._type == 'finite':
                    msg += ' ' * itv + '{:>17.10E}'.format(os[0].item()) + \
                           ' ' * itv + '{:>17.10E}'.format(os[1].item()) + \
                           ' ' * itv + '{:>17.10E}'.format(os[2].item())  # xyz coordinates
                msg += ' ' * itv + '{:>13.10f}'.format(ds[0].item()) + \
                       ' ' * itv + '{:>13.10f}'.format(ds[1].item()) + \
                       ' ' * itv + '{:>13.10f}'.format(ds[2].item())  # direction cosines
                msg += ' ' * itv + ' ' * 9 + '-'  # incident angle
                msg += ' ' * itv + ' ' * 15 + '-'  # path length
            else:
                msg += ' ' * itv + '{:>17.10E}'.format(os[0].item()) + \
                       ' ' * itv + '{:>17.10E}'.format(os[1].item()) + \
                       ' ' * itv + '{:>17.10E}'.format(os[2].item())  # xyz coordinates
                msg += ' ' * itv + '{:>13.10f}'.format(ds[0].item()) + \
                       ' ' * itv + '{:>13.10f}'.format(ds[1].item()) + \
                       ' ' * itv + '{:>13.10f}'.format(ds[2].item())  # direction cosines
                msg += ' ' * itv + '{:>13.10E}'.format(ai.item())  # incident angle
                msg += ' ' * itv + '{:>13.10E}'.format(ts.item())  # path length

            print(msg)

    def image_point(self, view, method='chief'):
        """to be deprecated"""
        if method == 'chief':
            ref_image_point = self.ref_rays_stacks_dict[view.item(), self.system.ref_wav.item()]['o'][-1][0]
        elif method == 'ideal':
            # default fov on y direction
            x = torch.tensor(0.)
            y = self.system.Distortion_Focal_Length * torch.tan(view.deg2rad())
            z = self.system[-1].origin[2]
            ref_image_point = torch.stack([x, y, z], dim=-1)
        elif method == 'centroid':
            # deprecated
            ref_ray = self.sample_ray_vig(wavelength=self.system.ref_wav, view=view)
            ref_ray = self.system.propagate(ref_ray)['ray']
            ref_image_point = ref_ray.o[ref_ray.valid_map].mean(0)
        else:
            raise Exception('method={} is not available!'.format(method))
        return ref_image_point

    # ====================================================================================
    # Image Quality
    # ====================================================================================
    def mtf(self, psf, image_delta, freq_max=None, freq_delta=None, show=False):
        if freq_max is None:
            freq_max = 1 / image_delta / 2
        if freq_delta is None:
            freq_delta = 1
        image_sampling = psf.shape[-1]
        freq_cut = 1 / image_delta / 2
        pad_points = int(freq_cut / freq_delta - image_sampling / 2) + 1
        num_points = int(freq_max / freq_delta) + 1
        index = int(pad_points + image_sampling / 2)

        psf_pad = torch.nn.functional.pad(psf, [pad_points, pad_points, pad_points, pad_points])

        T = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf_pad), dim=(-1, -2)))[..., index: index + num_points, index]
        S = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf_pad), dim=(-1, -2)))[..., index, index: index + num_points]
        freq = np.linspace(0, freq_max, num_points)

        if show:
            plt.plot(freq, T.detach().cpu(), '-', label='T')
            plt.plot(freq, S.detach().cpu(), '-.', label='S')
            plt.xlim(0)
            plt.ylim(0)
            plt.legend()

        return freq, T, S

    def relative_illumination(self, wavelength: torch.Tensor):
        """
        The Relative Illumination analysis computes the relative illumination
        as a function of radial field coordinate for a uniform Lambertian scene.

        Field_density[could be add later,now we just evaluate views]:
        The number of points along the radial field coordinate
        to compute the relative illumination for. Larger field densities yield smoother curves.

        Wavelength: Selects the wavelength for computation.
            Relative illumination is a monochromatic entity.
        Cite:
            wavelength must be involved in self.wavelengths
            assume aberration free: all ref rays intersect on the same image point
        """
        ri_list = []
        ref_rays = self.ref_rays[self.views[0].item(), wavelength.item()]
        ds = self.system.propagate(ref_rays)['d'][-1]
        m0 = ds[2, 1]
        for view in self.views:
            ref_rays = self.ref_rays[view.item(), wavelength.item()]
            ds = self.system.propagate(ref_rays)['d'][-1]
            ri = (ds[1, 1] - ds[2, 1]) * ds[3, 0] / (2 * m0 ** 2)
            ri_list.append(ri)
        return torch.stack(ri_list)

    def distortion(self, wavelength: torch.Tensor):
        d_list = []
        for view in self.views:
            ref_rays = self.ref_rays[view.item(), wavelength.item()]
            real_image_point = self.system.propagate(ref_rays)['o'][-1][0, 1]
            ideal_image_point = self.system.Distortion_Focal_Length * torch.tan(view.deg2rad())
            if real_image_point == ideal_image_point:
                distortion = torch.tensor(0.)
            else:
                distortion = (real_image_point - ideal_image_point) / ideal_image_point * 100
            d_list.append(distortion)
        return torch.stack(d_list)

    # ====================================================================================
    # Updates
    # ====================================================================================
    @torch.no_grad()
    def update_radius(self):
        """
        Fix the radius of pupil, update the radius of the all elements
        Ensuring the rays of all reference pupil rays could pass
        """
        os = []
        for view in self.views:
            for wavelength in self.wavelengths:
                os.append(self.ref_rays_stacks_dict[view.item(), wavelength.item()]['o'])
        for i, s in enumerate(self.system[1:]):
            if i + 1 == self.system.aperture_ind:
                continue
            r = torch.stack([length(o[i + 1][..., :2]) for o in os])
            s.r = r.max()

    def set_vignetting(self):
        for v in self.views:
            self.vig[v.item()] = self.calc_vignetting(v)

    def update_system(self):
        self.system._update_transforms()
        self.system._pre_calculation()
        # start_time = time.perf_counter()
        # self.ref_rays = self.system.aim_ref_rays(self.views, self.wavelengths, self.vig)
        # print(f"aim time cost:{time.perf_counter() - start_time}")
        # self.ref_rays_stacks_dict = self.ray_stacks_dict(self.ref_rays)
        # self.update_radius()

    def ray_stacks_dict(self, rays_dict):
        ray_stacks_dict = {}
        for view in self.views:
            for wavelength in self.wavelengths:
                ref_rays = rays_dict[view.item(), wavelength.item()]
                ray_stacks_dict[view.item(), wavelength.item()] = self.system.propagate(ref_rays)
        return ray_stacks_dict

    # ====================================================================================
    # optical feature collections
    # ====================================================================================
    def collect_all_rays_data(self, pupil_sampling=31):
        """
        首先，我们获得所有的光线数据
        如何存储数据：先按照视场、再按照波长进行分类
        最后返回数据字典[view][wavelength]{stacks + chief}
        """
        data_collection = {}
        for view in self.views:
            for wavelength in self.wavelengths:
                # rays sampling
                ray = self.sample_ray_vig(wavelength=wavelength, view=view, pupil_sampling=pupil_sampling)
                ref_ray = self.ref_rays[view.item(), wavelength.item()]
                data_collection[view.item(), wavelength.item()] = {'ray': self.system.propagate(ray, clip=True), 'ref_ray': self.system.propagate(ref_ray)}
        return data_collection

    def spots_collection(self, data_collection):
        """
        使用计算好的光线数据进行loss计算
        dim：view, wavelength 相对于主光线的rms loss
        """
        spot_views_wavs = []
        for view in self.views:
            spot_wavs = []
            image_point = data_collection[view.item(), self.system.ref_wav.item()]['ref_ray']['o'][-1][0, :2]
            for wavelength in self.wavelengths:
                rayo = data_collection[view.item(), wavelength.item()]['ray']['o'][-1][..., :2]
                valid = data_collection[view.item(), wavelength.item()]['ray']['valid'][-1]
                spot_wavs.append(torch.sqrt(torch.mean(torch.sum((rayo[valid] - image_point) ** 2, dim=-1))))
            spot_views_wavs.append(torch.stack(spot_wavs))
        return torch.stack(spot_views_wavs)

    def psf_collection(self, data_collection, image_sampling=31, image_delta=1e-3):
        """
        使用主光线像点计算的psf集合
        dim：view, wavelength
        """
        psf_views_wavs = []
        for view in self.views:
            psf_wavs = []
            image_point = data_collection[view.item(), self.system.ref_wav.item()]['ref_ray']['o'][-1][0]
            line_sample = torch.linspace(-1, 1, image_sampling) / 2 * image_delta * (image_sampling - 1)
            x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
            psf_coor = image_point + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
            for wavelength in self.wavelengths:
                wave_num = torch.tensor(2 * np.pi / wavelength.item())
                ray = data_collection[view.item(), wavelength.item()]['ray']['ray']
                # calculate the complex amplitude of each sampled image position
                r = psf_coor[:, :, None, :] - ray.o[None, None, ray.valid_map, :]
                # inner production for the final length, dim:x,y,ray,data
                dr = torch.einsum('ijkl,ijkl->ijk', ray.d[None, None, ray.valid_map, :], r)
                # complex amplitude
                amp = torch.einsum('ijk->ij', torch.exp(((ray.opd[None, None, ray.valid_map] + dr) * wave_num) * (0 + 1j)))
                psf = torch.flipud(torch.abs(amp).T ** 2)
                psf = psf / psf.sum()
                psf_wavs.append(psf)
            psf_views_wavs.append(torch.stack(psf_wavs))
        return torch.stack(psf_views_wavs)

    def opd_collection(self, data_collection):
        # 先找到像点，再构造出瞳
        opd_views_wavs = []
        for view in self.views:
            opd_wavs = []
            for wavelength in self.wavelengths:
                ref_ray = data_collection[view.item(), wavelength.item()]['ref_ray']['ray']
                ep_t = (self.system.global_exit_pupil_position - ref_ray.o[0, 2]) / ref_ray.d[0, 2]
                # 两种方式似乎差别不是很大
                # ep_t = -ref_ray.o[0, 2] / ref_ray.d[0, 2]
                ep_pos = ref_ray.o[0] + ep_t[..., None] * ref_ray.d[0]
                ep_roc = length(ref_ray.o[0] - ep_pos)
                theta_x = -torch.arctan(ref_ray.d[0, 1] / ref_ray.d[0, 2]).rad2deg()
                ep_theta = torch.stack([theta_x, torch.tensor(0.), torch.tensor(0.)])
                ep_surf = Spheric(roc=ep_roc, origin=ep_pos, theta_xyz=ep_theta)

                ray_ep = ep_surf.propagate(data_collection[view.item(), wavelength.item()]['ray']['ray'])
                ref_ep = ep_surf.propagate(ref_ray)
                opd = (ref_ep.opd[0] - ray_ep.opd) * ray_ep.valid_map / wavelength.item()
                sample = int(np.sqrt(len(ray_ep.opd)))
                opd_wavs.append(opd.view(sample, sample))
            opd_views_wavs.append(torch.stack(opd_wavs))
        return torch.stack(opd_views_wavs)

    def opd2mtf(self, opd_collection, freq=125):
        """
        test func
        自相关计算mtf
        """
        for i, view in enumerate(self.views):
            for j, wavelength in enumerate(self.wavelengths):
                opd = opd_collection[i, j]
                valid_map = opd != 0
                count = valid_map.sum()
                sample = opd.size(0)
                rms = torch.sqrt((opd ** 2).sum() / count - (opd.sum() / count) ** 2)
                field = (torch.exp(2j * opd * torch.pi) * valid_map).reshape(1, 1, sample, sample)
                freq_delta = 1 / ((sample - 1) * wavelength * self.system.fno).item()
                field_pad = torch.nn.functional.pad(field, (int(sample / 2), int(sample / 2), int(sample / 2), int(sample / 2)))
                mtf = torch.conv2d(field_pad, field.conj(), padding='same').abs() / count
                index = round(freq / freq_delta)
                mtfT = mtf[0, 0, sample - 1, sample - 1 + index]
                mtfS = mtf[0, 0, sample - 1 + index, sample - 1]
                print(f"index {index} T:{mtfT.item()} S:{mtfS.item()} rms:{rms}")
        raise NotImplementedError

    def wavefront_diff_test(self, data_collection):
        # opd = self.opd_collection(data_collection)
        def get_surf_points_normals(global_o):
            points = []
            normals = []
            for s, o in zip(self.system[1:], global_o):
                local_o = s.to_object.transform_point(o)
                normal = s.surface_normal(local_o[..., 0], local_o[..., 1])
                points.append(local_o)
                normals.append(normal)
            return torch.stack(points), torch.stack(normals)

        def get_gamma(incident, refract, wavelength):
            gammas = []
            for s, i, r in zip(self.system[1:], incident, refract):
                n1 = s.refractive_index_prev(wavelength)
                n2 = s.refractive_index(wavelength)
                gamma = n1 * torch.cos(i.deg2rad()) - n2 * torch.cos(r.deg2rad())
                gammas.append(gamma)
            return torch.stack(gammas)

        def get_diff_terms(rays_collection, wavelength):
            # stacks = {k: [] for k in ('o', 'd', 't', 'valid', 'in', 'out')}
            incident_angle = rays_collection['in']
            refracted_angle = rays_collection['out']
            global_o = rays_collection['o']
            local_o, normals = get_surf_points_normals(global_o)
            gammas = get_gamma(incident_angle, refracted_angle, wavelength)
            # 我们的法向是朝负方向的！
            # first we test displacement delta y
            perturbs = -normals[..., 1] * 1e-3
            diff_terms = perturbs * gammas
            return diff_terms

        wd_views_wavs = []
        for view in self.views:
            wd_wavs = []
            for wavelength in self.wavelengths:
                rays_collection = data_collection[view.item(), wavelength.item()]['ray']
                ref_ray_collection = data_collection[view.item(), wavelength.item()]['ref_ray']
                diff_terms = get_diff_terms(rays_collection, wavelength)
                ref_diff_terms = get_diff_terms(ref_ray_collection, wavelength)[:, 0].unsqueeze(-1)
                wd_wavs.append(ref_diff_terms-diff_terms)
                # wd_wavs.append(-diff_terms)
            wd_views_wavs.append(torch.stack(wd_wavs))

        return torch.stack(wd_views_wavs)

    def get_tolerance_coefficients(self, opd_collection, wavefront_diff):
        def opd2otf(opd_collection, freq=125):
            otf_views_wavs = []
            for i, view in enumerate(self.views):
                otf_wavs = []
                for j, wavelength in enumerate(self.wavelengths):
                    opd = opd_collection[i, j]
                    valid_map = opd != 0
                    count = valid_map.sum()
                    sample = opd.size(0)
                    # rms = torch.sqrt((opd ** 2).sum() / count - (opd.sum() / count) ** 2)
                    field_c = (torch.exp(2j * opd * torch.pi) * valid_map).reshape(1, 1, sample, sample)
                    field_c_pad = torch.nn.functional.pad(field_c, (int(sample / 2), int(sample / 2), int(sample / 2), int(sample / 2)))
                    otf_c = torch.conv2d(field_c_pad, field_c.conj(), padding='same') / count

                    # turn into rad unit
                    wd = wavefront_diff[i, j].reshape(-1, sample, sample) * valid_map / wavelength * 2 * torch.pi
                    tes0 = wd.detach().cpu().numpy()
                    field_b = wd * field_c
                    tes1 = field_b.abs().detach().cpu().numpy()
                    field_b_pad = torch.nn.functional.pad(field_b, (int(sample / 2), int(sample / 2), int(sample / 2), int(sample / 2)))
                    otf_b = torch.conv2d(field_b_pad, field_b.permute(1, 0, 2, 3).conj(), padding='same', groups=field_b.shape[1]) / count
                    tes2 = otf_b.abs().detach().cpu().numpy()

                    field_a = wd**2 * field_c
                    field_a_pad = torch.nn.functional.pad(field_a, (int(sample / 2), int(sample / 2), int(sample / 2), int(sample / 2)))
                    otf_a = torch.conv2d(field_a_pad, field_a.permute(1, 0, 2, 3).conj(), padding='same', groups=field_a.shape[1]) / count

                    field_d = torch.exp(1j*wd)*field_c
                    field_d_pad = torch.nn.functional.pad(field_d, (int(sample / 2), int(sample / 2), int(sample / 2), int(sample / 2)))
                    otf_d = torch.conv2d(field_d_pad, field_d.permute(1, 0, 2, 3).conj(), padding='same', groups=field_a.shape[1]) / count

                    freq_delta = 1 / ((sample - 1) * wavelength * self.system.fno).item()
                    index = round(freq / freq_delta)
                    # mtfT = mtf[0, :, sample - 1, sample - 1 + index]
                    # mtfS = mtf[0, :, sample - 1 + index, sample - 1]
                    coeff = torch.stack((otf_d, otf_a, otf_b, torch.broadcast_to(otf_c, otf_a.shape)), dim=-1)
                    # coeff = torch.stack((-0.5 * otf_a, 1j * otf_b, torch.broadcast_to(otf_c, otf_a.shape)), dim=-1)
                    coeff_T = coeff[0, :, sample - 1, sample - 1 + index]
                    coeff_S = coeff[0, :, sample - 1 + index, sample - 1]
                    otf_wavs.append(torch.stack([coeff_T, coeff_S]))
                otf_views_wavs.append(torch.stack(otf_wavs))
            return torch.stack(otf_views_wavs)
        # shape: view, wavelength, T/S, surface, coeff_a/b/c
        res = opd2otf(opd_collection)
        return res

    def coeff2mtf(self, coeff_collection):
        """
        test func for MTF = a*p^2 + b*p + c
        """
        raise NotImplementedError

    # ====================================================================================
    # Applications (test more)
    # ====================================================================================
    def tolerance_diff(self, pupil_sampling):
        """
        计算了一个感度loss,保留了计算图用于回传
        """
        # --------step1--------
        # 先进行追迹分析 记录中间变量
        element = torch.tensor(0.).repeat(len(self.system) - 2)
        freq = 125

        # torch.set_anomaly_enabled(True)
        def loss_func_decenter(shift_y):
            for i in range(len(self.system))[1:-1]:
                self.system[i].shift = torch.stack([torch.zeros_like(shift_y[i - 1]), shift_y[i - 1], torch.zeros_like(shift_y[i - 1])])
            # self.update_system()
            self.system._update_transforms()
            # self.update_system()
            dc = self.collect_all_rays_data(pupil_sampling=pupil_sampling)
            psfs = self.psf_collection(dc)
            _, T, S = self.mtf(psfs, 1e-3)
            mtf_125 = []
            for j, v in enumerate(self.views):
                print(f"f{v.item()} MTF 125 LPMM: T{T[j, 0, freq].item()} S{S[j, 0, freq].item()}")
                # mtf_125.append(T[j, 0, 125] + S[j, 0, 125])
                mtf_125.append(torch.stack([T[j, 0, freq], S[j, 0, freq]]))
            return torch.stack(mtf_125)

        def loss_func_tilt(tilt_x):
            for i in range(len(self.system))[1:-1]:
                self.system[i].theta_xyz = torch.stack([tilt_x[i - 1], torch.zeros_like(tilt_x[i - 1]), torch.zeros_like(tilt_x[i - 1])])
            # self.update_system()
            self.system._update_transforms()
            # self.update_system()
            dc = self.collect_all_rays_data(pupil_sampling=pupil_sampling)
            psfs = self.psf_collection(dc)
            _, T, S = self.mtf(psfs, 1e-3)
            mtf_125 = []
            for j, v in enumerate(self.views):
                print(f"f{v.item()} MTF 125 LPMM: T{T[j, 0, freq].item()} S{S[j, 0, freq].item()}")
                # mtf_125.append(T[j, 0, 125] + S[j, 0, 125])
                mtf_125.append(torch.stack([T[j, 0, freq], S[j, 0, freq]]))
            return torch.stack(mtf_125)

        def loss_func_thickness(shift_z):
            for i in range(len(self.system))[1:-1]:
                self.system[i].shift = torch.stack([torch.zeros_like(shift_z[i - 1]), torch.zeros_like(shift_z[i - 1]), shift_z[i - 1]])
            # self.update_system()
            self.system._update_transforms()
            # self.update_system()
            dc = self.collect_all_rays_data(pupil_sampling=pupil_sampling)
            psfs = self.psf_collection(dc)
            _, T, S = self.mtf(psfs, 1e-3)
            mtf_125 = []
            for j, v in enumerate(self.views):
                print(f"f{v.item()} MTF 125 LPMM: T{T[j, 0, freq].item()} S{S[j, 0, freq].item()}")
                # mtf_125.append(T[j, 0, 125] + S[j, 0, 125])
                mtf_125.append(torch.stack([T[j, 0, freq], S[j, 0, freq]]))
            return torch.stack(mtf_125)

        # --------step2--------
        # sd = torch.autograd.functional.hessian(loss_func, shift_y)
        # ss = sd.diag()
        # return ss
        #
        from torch.func import jacrev, jacfwd
        # fd = jacrev(loss_func_decenter)(element)
        fd = jacrev(loss_func_tilt)(element)
        # fd = jacrev(loss_func_thickness)(element)
        return fd

    def tolerance_visualization(self):
        # shape (view,Surface)
        ssty = self.tolerance_diff(128)
        data = ssty / opticsmodel.length(ssty).unsqueeze(-1)
        l1 = opticsmodel.length(data)
        l2 = opticsmodel.length(ssty)

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        # lines plot
        colors_list = 'bgrymck'
        for i, s in enumerate(ssty):
            # ax.plot(np.asarray(s.cpu().detach()), color=colors_list[i], label="fov {}".format(self.views[i].item()))
            print(f"fov {self.views[i].item()} T: {np.asarray(s[0].cpu().detach())*1e-3}")
            print(f"fov {self.views[i].item()} S: {np.asarray(s[1].cpu().detach())*1e-3}")
            ax.plot(np.asarray(s[0].cpu().detach()/30), color=colors_list[i], label="fov {} T".format(self.views[i].item()))
            ax.plot(np.asarray(s[1].cpu().detach()/30), color=colors_list[i], linestyle='--', label="fov {} S".format(self.views[i].item()))
        ax.legend()
        ax.set_xlabel("different surface")

        # stack plot
        # sensitivity_views = np.asarray(sensitivity_views).T
        # sensitivity_views /= sensitivity_views.sum(axis=0)
        # ax.stackplot(["fov:{}".format(v) for v in views], sensitivity_views)
        # ax.set_xlabel("different field of view")
        # ax.set_ylabel("different surface")

        ax.set_title("sensitivity analysis TILT")
        ax.autoscale(enable=True, axis='both', tight=True)
        plt.savefig("double gauss TILT 125lp grads2mtf")
        plt.show()
        # plt.tight_layout()
        # plt.axis('tight')
        return fig, ax, sensitivity_views

    # ====================================================================================
    # Depreciated later
    # ====================================================================================
    def sample_gausslet(self, wavelength=DEFAULT_WAVELENGTH, view=torch.tensor(0.), coeff=50):
        """deprecated, edit later"""
        # update the field-of-view of system
        self.system[0].update(view)
        # maximum radius input
        xs, ys = self.calc_vignetting(view, wavelength)
        # 实际上因为渐晕裁切的关系，可能是一个椭圆切去顶部,所以适当增大一下这个椭圆
        x_mean, y_mean = xs.mean(), ys.mean()
        x_r = (xs.max() - xs.min()) * 0.55
        y_r = (ys.max() - ys.min()) * 0.55

        pi = torch.pi
        width = wavelength * coeff
        theta = wavelength / pi / width
        fov = view.deg2rad()
        # OF = 1.7 经验数值
        y_resolution = int(2 * y_r / width * 1.7)
        x_resolution = int(2 * x_r / width * 1.7)
        # 确定倾斜波前面上的采样点
        y_ = torch.linspace(y_mean - y_r, y_mean + y_r, int(y_resolution / torch.cos(fov)))
        x_ = torch.linspace(x_mean - x_r, x_mean + x_r, x_resolution)
        x, y = torch.meshgrid(x_, y_, indexing='ij')
        z = -y * torch.tan(fov)
        dx = -torch.zeros_like(x)
        dy = -torch.ones_like(y) * torch.tan(fov)
        wavefront = torch.stack((x, y, z), dim=-1)
        wavefront_norm = normalize(torch.stack((-dx, -dy, torch.ones_like(dx)), dim=-1))

        gausslet_collection = gen_decomp(wavelength, width, theta, wavefront, wavefront_norm)

        return gausslet_collection

    def psf_gausslet(self, image_sampling, image_delta, coeff=500, wavelength=None, view=None):
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths) // 2]
        if view is None:
            view = self.views[-1]

        # gausslets sampling
        gausslets = self.sample_gausslet(wavelength=wavelength, view=view, coeff=coeff)
        # initialize the sample complex amplitude mat on image plane
        # with torch.no_grad():
        #     image_point = self.get_real_image_point(view, ref_wav=wavelength)
        image_point = self.image_point(view)
        line_sample = torch.linspace(-1, 1, image_sampling) / 2 * image_delta * (image_sampling - 1)
        x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        psf_coor = image_point + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
        for s in self.system[1:-1]:
            gausslets.base_ray = s.propagate(gausslets.base_ray)['ray']
            gausslets.waist_rayx = s.propagate(gausslets.waist_rayx)['ray']
            gausslets.waist_rayy = s.propagate(gausslets.waist_rayy)['ray']
            gausslets.div_rayx = s.propagate(gausslets.div_rayx)['ray']
            gausslets.div_rayy = s.propagate(gausslets.div_rayy)['ray']
        valid_base = gausslets.base_ray.valid_map & gausslets.waist_rayx.valid_map & \
                     gausslets.waist_rayy.valid_map & gausslets.div_rayx.valid_map & \
                     gausslets.div_rayy.valid_map
        gausslets_out = gausslets.clip(valid_base)
        gausslets_out.opd = gausslets_out.base_ray.opd

        psf = torch.abs(gausslets_out.calculate_map(psf_coor)) ** 2
        psf = torch.flipud(psf.T / psf.sum())
        # psf = psf.T
        return psf

    def eval_surf_distance(self, sample=256):
        d_list = []
        s_prev = self.system[1]
        for s in self.system[2:]:
            r = torch.minimum(s_prev.r, s.r)
            r_sample = torch.linspace(0, r, sample)
            d = s.surface_with_offset(r_sample, torch.zeros_like(r_sample)) - s_prev.surface_with_offset(r_sample, torch.zeros_like(r_sample))
            d_list.append(d)
            s_prev = s
        return torch.stack(d_list)

    def eval_surf_ratio(self,sample=256):
        ratio_list = []
        s_prev = self.system[1]
        for s in self.system[2:]:
            if s_prev.Material.nd > 1.0003:
                r = torch.minimum(s_prev.r, s.r)
                r_sample = torch.linspace(0, r, sample)
                d = s.surface_with_offset(r_sample, torch.zeros_like(r_sample)) - s_prev.surface_with_offset(r_sample,
                                                                                                         torch.zeros_like(
                                                                                                             r_sample))
                ratio_list.append(d/r)
            s_prev = s
        return torch.stack(ratio_list)



    def save_zmx(self, zmx_file):

        zmx = open(zmx_file, 'w')
        head_str = f"""
VERS 190513 25 123457 L123457
MODE SEQ
NAME
UNIT MM X W X CM MR CPMM
FLOA
GCAT SCHOTT APEL MISC CDGM HOYA PLASTIC-EP9000
RAIM 0 2 1 1 0 0 0 0 0 1
FTYP 0 0 {len(self.views)} {len(self.wavelengths)} 0 0 0 {len(self.views)}"""
        zmx.writelines(head_str)

        xfln = "XFLN"
        yfln = "YFLN"
        for i, view in enumerate(self.views):
            xfln += " "
            xfln += str(0)
            yfln += " "
            yfln += str(view.item())
        zmx.writelines(f"""
{xfln}""")
        zmx.writelines(f"""
{yfln}""")

        for i, wave in enumerate(self.wavelengths):
            wave_str = f"""
WAVM {i + 1} {wave.item()*1e3} 1"""
            zmx.writelines(wave_str)
        zmx.writelines("""
PWAV 1""")

        obj_str = f"""
SURF 0
    TYPE STANDARD
    FIMP
    CURV 0.0
    DISZ INFINITY
    DIAM 0"""
        zmx.writelines(obj_str)

        for i, surf in enumerate(self.system[1:-1]):
            if i + 1 == self.system.aperture_ind:
                surf_str = f"""
SURF {i + 1}
    STOP
    TYPE EVENASPH
    FIMP"""
            else:
                surf_str = f"""
SURF {i + 1}
    TYPE EVENASPH
    FIMP"""
            zmx.writelines(surf_str)

            surf_str = f"""            
    CURV {surf.c.item()}
    DISZ {surf.d.item()}
    CONI {surf.k.item()}
    DIAM {surf.r.item()}"""
            zmx.writelines(surf_str)

            for ai in range(surf.maximum_term):
                # ! sqrt ai -> ai
                surf_str = f"""
    PARM {ai + 2} {eval("surf.ai_{}.item()".format(2 * ai + 4))}"""
                zmx.writelines(surf_str)

            if surf.Material.name != "vacuum":
                zmx.writelines(f"""
    GLAS {surf.Material.name.upper()}""")

        img_str = f"""
SURF {len(self.system) - 1}
    TYPE STANDARD
    FIMP
    CURV 0.0
    DISZ 0
    DIAM {self.system[-1].r.item()}"""
        zmx.writelines(img_str)

    # ====================================================================================
    # Test version methods
    # ====================================================================================
    def optical_path_difference_fan(self, rays_num, wavelength, view, surface):
        """
        calculate the optical path difference of the rays with
        rays sampling on x and y direction [rays_num]
        wavelength [wavelength]
        field-of-view [view]
        on the surface index [surface]
        return the curve on x and y direction
        """
        raise NotImplementedError

    def wavefront_map(self, sampling=101, wavelength=None, view=None, surface=-1, show=True):
        """
        calculate the wavefront map of system with
        rays sampling [rays_num]
        wavelength [wavelength]
        field-of-view [view]
        on the surface index [surface]
        return the 2D wavefront map
        """
        raise NotImplementedError

    def mtf_through_focus(self, pupil_sampling, image_sampling, image_delta, frequency,
                          delta_focus, steps, wavelength=None, view=None):
        """
        Computes the diffraction modulation transfer function (MTF) data
        using Huygens PSFs and displays the data as a function of delta focus.

        pupil_sampling: the size of the grid of rays to trace to perform the computation.
        image_sampling: The size of the grid of points on which to compute the diffraction image intensity.
        image_delta: The distance in micrometers between points in the image grid.
        frequency: The spatial frequency (cycle per micrometers).
        wavelength: The wavelength number to be used in the calculation.
        view: field of view.
        delta_focus: delta focus is the ± Z-axis range of the plot in micrometers.
        steps: The number of focal planes at which the data is computed.
        todo: through focus psf, such as edit ep_surf for different defocus; zrr: too tired to clean code, try later
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths) // 2]
        if view is None:
            view = self.views[-1]

        # rays sampling
        ray, ray_chief = self.image_cache(
            pupil_sampling, view, wavelength, 'fibonacci')

        # optical path calculation with propagate
        valid = torch.ones_like(ray.t).bool()
        valid_chief = torch.ones_like(ray_chief.t).bool()
        ## if input object plane is infinite and tilt       ##
        ## we assume the optical path of the chief ray is 0 ##
        op = torch.einsum('ij,ij->i', ray.o - ray_chief.o, ray_chief.d)
        op_chief = torch.einsum(
            'ij,ij->i', ray_chief.o - ray_chief.o, ray_chief.d)

        # propagate to the last surface of lens (before image plane)
        for s in self.system[1:None]:
            valid, ray = s.propagate(ray, valid)
            op += ray.t * s.refractive_index_prev(wavelength)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)
            op_chief += ray_chief.t * s.refractive_index_prev(wavelength)

        # calculate the intersection of ray and optical axis to judge the exit pupil
        # NOTE: Only y direction cosine calculation is supported

        if view == 0:
            self.system._paraxial_info(self.views)
            ep_roc = - self.system.Exit_Pupil_Position
        else:
            ep_roc = ray_chief.o[..., 1] / ray_chief.d[..., 1]
            ep_roc = ep_roc.squeeze()

        # origin_exit_pupil = ray_chief.o - t_ep * ray_chief.d
        ep_distance = self.system[-1].d - ep_roc
        ep_distance_prev = -ep_roc
        ep_radius = 2. * ep_roc

        ep_surf = Spheric(roc=ep_roc,
                          conic=None,
                          radius=ep_radius,
                          distance=ep_distance,
                          distance_prev=ep_distance_prev,
                          distance_after=-ep_distance_prev,
                          material='vacuum',
                          material_prev='vacuum'
                          )  # exit pupil surface related to the center of sensor
        # shift rays on image plane to the center of sensor
        op = op[valid]
        o_final = torch.zeros_like(ray.o[valid])
        o_final[..., 0:2] = ray.o[valid][..., 0:2] - ray_chief.o[..., 0:2]
        o_final[..., 2] = ray.o[valid][..., 2]
        # form the ray backpropagate to the sensor plane
        ray_shift = Ray(o=o_final, d=ray.d[valid], wavelength=ray.wavelength)
        ray_ep = ep_surf.propagate(ray_shift, torch.ones_like(ray_shift.t).bool())[1]
        ray_ep.o[..., 0:2] += ray_chief.o[..., 0:2]
        op += ray_ep.t

        wave_num = torch.tensor(2 * np.pi / wavelength.item())  # 10000
        MTF_T_through_focus = torch.zeros(steps)
        MTF_S_through_focus = torch.zeros(steps)
        # traverse the defocus position along the +-Z-axis
        z_idx = 0
        for z_delta in torch.linspace(-delta_focus, delta_focus, steps):
            # calculate the optical path difference of the sampled point on image
            # initialize the sample complex amplitude mat on image plane
            intensity_map = torch.zeros((image_sampling, image_sampling))

            # calculate the complex amplitude of each sampled image position
            line_sample = torch.linspace(- int((intensity_map.shape[0] - 1) / 2),
                                         int((intensity_map.shape[0] - 1) / 2),
                                         image_sampling) * image_delta
            x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
            # calculate the chief ray position after sensor plane has shifted
            t_delta = z_delta / ray_chief.d[..., 2]
            o_delta = ray_chief.o + t_delta * ray_chief.d
            rel_coor = o_delta + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
            r = rel_coor[:, :, None, :] - ray_ep.o[None, None, ...]
            # inner production for the final length
            dr = torch.einsum('ijkl,ijkl->ijk', ray_ep.d[None, None, ...], r)
            # complex amplitude
            amp = torch.einsum(
                'ijk->ij', torch.exp(((op[None, None, :] + dr) * wave_num) * (0 + 1j)))
            psf = torch.real(amp * torch.conj(amp)).permute(1, 0)
            psf = psf / psf.sum()

            num_points = np.int64(
                np.ceil(
                    np.log2(
                        image_sampling /
                        32) +
                    1) *
                50)
            pad_points = np.int64(num_points - image_sampling / 2)
            psf = torch.nn.functional.pad(
                psf, [pad_points, pad_points, pad_points, pad_points])
            # MTF in T and S directions
            T = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf)))[
                :, num_points][num_points:]
            S = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf)))[
                num_points, :][num_points:]
            freq = torch.linspace(0, 1 / image_delta / 2, num_points)
            # find the position for interpolation
            freq_idx = (freq < frequency).sum()  # the number of true
            T_itp = (T[freq_idx - 1] * (freq[freq_idx] - frequency) + T[freq_idx] * (frequency - freq[freq_idx - 1])) / \
                    (freq[freq_idx] - freq[freq_idx - 1])
            S_itp = (S[freq_idx - 1] * (freq[freq_idx] - frequency) + S[freq_idx] * (frequency - freq[freq_idx - 1])) / \
                    (freq[freq_idx] - freq[freq_idx - 1])

            MTF_T_through_focus[z_idx] = T_itp
            MTF_S_through_focus[z_idx] = S_itp
            z_idx += 1

        return MTF_T_through_focus, MTF_S_through_focus
