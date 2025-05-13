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
    # Updates
    # ====================================================================================
    def set_vignetting(self):
        for v in self.views:
            self.vig[v.item()] = self.calc_vignetting(v)

    def update_system(self):
        self.system._update_transforms()
        self.system._pre_calculation()


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


