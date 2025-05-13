import json
import os
from itertools import groupby


from .conjugates import *
from .surfaces import *
from .basics import DEFAULT_WAVELENGTH, Ray  # 587nm, 486nm, 656nm
from functools import reduce


class System(nn.Module):
    """
    The Lensgroup (consisted of multiple optical surfaces) is mounted on a rod, whose
    origin is `origin`. The Lensgroup has full degree-of-freedom to rotate around the
    x/y axes, with the rotation angles defined as `theta_x`, `theta_y`, and `theta_z` (in degree).

    In Lensgroup's coordinate (i.e. object frame coordinate), surfaces are allocated
    starting from `z = 0`. There is an additional, comparatively small 3D origin shift
    (`shift`) between the surface center (0,0,0) and the origin of the mount, i.e.
    shift + origin = lensgroup_origin.
    """

    def __init__(self, config_file, ref_wav=DEFAULT_WAVELENGTH):
        super(System, self).__init__()
        self.config = config_file
        if os.path.splitext(config_file)[-1] == '.json':
            self.load_file_from_json(config_file)
        else:
            raise ValueError('pls using json file')
        self.ref_wav = torch.tensor(ref_wav)
        self.APERTURE_TOLERENCE = 1e-10
        self.APERTURE_ITER_MAX = 10

    def __getitem__(self, idx):
        return self.surfaces[idx]

    def __len__(self):
        return len(self.surfaces)

    # ====================================================================================
    # Initialization
    # ====================================================================================
    def load_file_from_json(self, file_path):
        self.surfaces = nn.ModuleList([])
        d_global = 0.
        # propagate_forward = True
        with open(file_path) as file:
            lens_dict = json.load(file)
        file.close()
        self.var_dict = {}

        for item in lens_dict:
            # split the string and the numbers
            itemize = [''.join(list(g)) for k, g in groupby(item, key=lambda x: x.isdigit())]

            if itemize[0] == 'Description':
                self.LensName = lens_dict['Description']
                continue

            elif itemize[0] == 'OBJECT':
                self.surfaces.append(InfiniteConjugate())
                material_prev = lens_dict[item]['material']

            elif itemize[0] == 'Standard' or itemize[0] == 'Binary' or itemize[0] == 'STOP':
                if 'coeff-terms' in lens_dict[item].keys() and lens_dict[item]['coeff-terms'] is not None:
                    self.surfaces.append(Binary2(index=lens_dict[item]['index'],
                                                 roc=lens_dict[item]['roc'],
                                                 distance=d_global,
                                                 distance_prev=d_prev,
                                                 distance_after=lens_dict[item]['distance'],
                                                 material=lens_dict[item]['material'],
                                                 material_prev=material_prev,
                                                 radius=lens_dict[item]['radius'],
                                                 conic=lens_dict[item]['conic'],
                                                 ai=lens_dict[item]['ai-terms'],
                                                 diff_order=lens_dict[item]['coeff-terms'][0],
                                                 norm_r=lens_dict[item]['coeff-terms'][1],
                                                 coeff_pi=lens_dict[item]['coeff-terms'][2:],
                                                 variables=lens_dict[item]['variable'] if 'variable' in lens_dict[item].keys() else [],
                                                 origin=lens_dict[item]['origin'] if 'origin' in lens_dict[item].keys() else [],
                                                 shift=lens_dict[item]['shift'] if 'shift' in lens_dict[item].keys() else np.zeros(3),
                                                 theta_xyz=lens_dict[item]['theta_xyz'] if 'theta_xyz' in lens_dict[item].keys() else np.zeros(3)
                                                 ))
                elif 'ai-terms' in lens_dict[item].keys() and lens_dict[item]['ai-terms'] is not None:
                    self.surfaces.append(Aspheric(index=lens_dict[item]['index'],
                                                  roc=lens_dict[item]['roc'],
                                                  distance=d_global,
                                                  distance_prev=d_prev,
                                                  distance_after=lens_dict[item]['distance'],
                                                  material=lens_dict[item]['material'],
                                                  material_prev=material_prev,
                                                  radius=lens_dict[item]['radius'],
                                                  conic=lens_dict[item]['conic'],
                                                  ai=lens_dict[item]['ai-terms'],
                                                  variables=lens_dict[item]['variable'] if 'variable' in lens_dict[item].keys() else [],
                                                  origin=lens_dict[item]['origin'] if 'origin' in lens_dict[item].keys() else [],
                                                  shift=lens_dict[item]['shift'] if 'shift' in lens_dict[item].keys() else np.zeros(3),
                                                  theta_xyz=lens_dict[item]['theta_xyz'] if 'theta_xyz' in lens_dict[item].keys() else np.zeros(3)
                                                  ))
                else:
                    self.surfaces.append(Spheric(index=lens_dict[item]['index'],
                                                 roc=lens_dict[item]['roc'],
                                                 distance=d_global,
                                                 distance_prev=d_prev,
                                                 distance_after=lens_dict[item]['distance'],
                                                 material=lens_dict[item]['material'],
                                                 material_prev=material_prev,
                                                 radius=lens_dict[item]['radius'],
                                                 conic=lens_dict[item]['conic'],
                                                 variables=lens_dict[item]['variable'] if 'variable' in lens_dict[item].keys() else [],
                                                 origin=lens_dict[item]['origin'] if 'origin' in lens_dict[item].keys() else [],
                                                 shift=lens_dict[item]['shift'] if 'shift' in lens_dict[item].keys() else np.zeros(3),
                                                 theta_xyz=lens_dict[item]['theta_xyz'] if 'theta_xyz' in lens_dict[item].keys() else np.zeros(3)
                                                 ))

                if itemize[0] == 'STOP':
                    self.aperture_ind = len(self.surfaces) - 1  # surfaces including object
                    # self[self.aperture_ind].clip = True

            elif itemize[0] == 'IMAGE':
                self.surfaces.append(Spheric(index=lens_dict[item]['index'],
                                             roc=lens_dict[item]['roc'],
                                             distance=d_global,
                                             distance_prev=d_prev,
                                             distance_after=lens_dict[item]['distance'],
                                             material=lens_dict[item]['material'],
                                             material_prev=material_prev,
                                             radius=lens_dict[item]['radius'] * 1.1,  # to avoid clip rays on sensor, modify later
                                             conic=lens_dict[item]['conic'],
                                             variables=lens_dict[item]['variable'] if 'variable' in lens_dict[item].keys() else [],
                                             origin=lens_dict[item]['origin'] if 'origin' in lens_dict[item].keys() else [],
                                             shift=lens_dict[item]['shift'] if 'shift' in lens_dict[item].keys() else np.zeros(3),
                                             theta_xyz=lens_dict[item]['theta_xyz'] if 'theta_xyz' in lens_dict[item].keys() else np.zeros(3)
                                             ))
            # saving variables
            if lens_dict[item]['variable']: self.var_dict[len(self.surfaces) - 1] = lens_dict[item]['variable']
            material_prev = 'vacuum' if lens_dict[item]['material'].upper() == 'MIRROR' else lens_dict[item]['material']
            d_prev = lens_dict[item]['distance']
            d_global += lens_dict[item]['distance']

    # update system data after change system
    def _update_transforms(self):
        """
        根据序列模式计算坐标变换关系,
        但是这样子会将shift的梯度通过序列计算连接起来，
        所以先确定点的位置和方向，然后detach*d, work!!
        """
        zero = torch.tensor(0.)
        origin = torch.stack([zero, zero, zero])
        direction = torch.stack([zero, zero, torch.tensor(1.)])
        world_origin = origin
        for i in range(len(self))[1:]:
            self[i].update_transform(world_origin)
            world_origin = self[i].to_world.transform_point(origin) + self[i].to_world.transform_vector(direction) * self[i].d
        # world_origin = origin
        # for i in range(len(self))[1:]:
        #     self[i].update_transform(world_origin.clone())
        #     world_origin += torch.stack([zero, zero, self[i].d])

    def _pre_calculation(self):
        """
        update system first order data
        """
        paraxial_matrixs = [s.paraxial_matrix(wavelength=self.ref_wav) for s in self[1:]]
        # check stop at first surface
        if self.aperture_ind == 1:
            self.entrance_pupil_position = self[self.aperture_ind].origin[2]
            self.entrance_pupil_radius = self[self.aperture_ind].r
        else:
            abcd2stop = reduce((lambda x, y: y @ x), paraxial_matrixs[:self.aperture_ind - 1])
            self.entrance_pupil_position = abcd2stop[0, 1] / abcd2stop[0, 0]
            self.entrance_pupil_radius = self[self.aperture_ind].r / abcd2stop[0, 0]
        abcd2last = reduce((lambda x, y: y @ x), paraxial_matrixs[self.aperture_ind - 1:-1])
        self.exit_pupil_position = -abcd2last[0, 1] / abcd2last[1, 1]
        self.exit_pupil_radius = self[self.aperture_ind].r * (abcd2last[0, 0] - abcd2last[0, 1] * abcd2last[1, 0] / abcd2last[1, 1])
        abcd_all = reduce((lambda x, y: y @ x), paraxial_matrixs[:-1])
        self.effl = -1 / abcd_all[1, 0]
        self.fno = self.effl / (2 * self.entrance_pupil_radius)

        # calculate distortion focal length
        o = torch.stack([torch.tensor(0.), torch.tensor(0.), self.Entrance_Pupil_Position], dim=-1)
        angle = torch.tensor(1E-16)
        d = torch.stack([torch.tensor(0.), torch.sin(angle), torch.cos(angle)], dim=-1)
        ray = Ray(o.unsqueeze(0), d.unsqueeze(0), wavelength=self.ref_wav)  # check default wavelength
        ray = self.propagate(ray)['ray']
        self.dfl = ray.o[0, 1] / torch.tan(angle)

    def get_variables(self):
        params_list = []
        params_name = []
        for k, vs in self.var_dict.items():
            for v in vs:
                if v == 'ai':  # only for aspheric
                    max_term = self[k].maximum_term
                    for i in range(max_term):
                        # exec(f"self[{k}].{v}_{2*(i+1)+2}.requires_grad_(True)")
                        exec(f"params_list.append(self[{k}].{v}_{2 * (i + 1) + 2})")
                        exec(f"params_name.append(\'{''.join([str(k), '_', v, str(2 * (i + 1) + 2)])}\')")
                else:
                    # exec(f"self[{k}].{v}.requires_grad_(True)")
                    exec(f"params_list.append(self[{k}].{v})")
                    exec(f"params_name.append(\"{''.join([str(k), '_', v])}\")")
        return params_list, params_name

    def set_variables(self, params):
        index = 0
        for k, vs in self.var_dict.items():
            for v in vs:
                if v == 'ai':  # only for aspheric
                    max_term = self[k].maximum_term
                    for i in range(max_term):
                        exec(f"self[{k}].{v}_{2 * (i + 1) + 2}=params[{index + i}]")
                    index += max_term
                else:
                    exec(f"self[{k}].{v}=params[{index}]")
                    index += 1

    # ====================================================================================
    # Rays aiming
    # ====================================================================================
    @torch.no_grad()
    def aim_ray(self, Px, Py, fov, azimuth, wavelength=DEFAULT_WAVELENGTH):
        """
        Aiming the position of the normalized position [Px, Py] on pupil plane
        return the cls Ray on the entrance pupil plane
        """
        # update the field-of-view of system
        self[0].update(fov, azimuth)
        # generate the initial ray
        x_init = (self.entrance_pupil_radius * Px)
        y_init = (self.entrance_pupil_radius * Py)
        z = self.entrance_pupil_position
        if self[0]._type == 'infinite':
            # infinite case: fixed angle
            o = torch.stack([x_init, y_init, z], dim=-1).unsqueeze(0)
            d = self[0].map().unsqueeze(0)
        elif self[0]._type == 'finite':
            # finite case: fixed object position
            o = torch.stack([x_init, y_init, z], dim=-1).unsqueeze(0)
            d = normalize(torch.tensor([0., 0., 0.]) - self[0].o).unsqueeze(0)

        # form the ray
        ray_init = Ray(o, d, wavelength=wavelength)
        # optimize the ray's path to let it pass through the center of stop
        it = 0
        scale = self.entrance_pupil_radius / self[self.aperture_ind].r
        # calcuate the initial residual
        ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind + 1)['ray']
        ray_stop_local = self[self.aperture_ind].to_object.transform_ray(ray_stop).o
        residualx = (Px * self[self.aperture_ind].r - ray_stop_local[0, 0])
        residualy = (Py * self[self.aperture_ind].r - ray_stop_local[0, 1])

        while (residualx.abs() + residualy.abs() > self.APERTURE_TOLERENCE).any() and (it < self.APERTURE_ITER_MAX):
            """
            测了一下还是比较线性的，梯度大致都在0.5左右，所以近似线性更新就好了
            """
            # 这一段需要residual.abs_()
            # grad_x, grad_y = torch.func.grad(get_residual, argnums=(0, 1))(x_init, y_init)
            # x_init = torch.where(grad_x != 0, x_init - residual_x / grad_x, x_init)
            # y_init = torch.where(grad_y != 0, y_init - residual_y / grad_y, y_init)
            x_init += residualx * scale
            y_init += residualy * scale

            o = torch.stack([x_init, y_init, z], dim=-1).unsqueeze(0)
            ray_init = Ray(o, d, wavelength=wavelength)
            ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind + 1)['ray']
            ray_stop_local = self[self.aperture_ind].to_object.transform_ray(ray_stop).o
            residualx = (Px * self[self.aperture_ind].r - ray_stop_local[0, 0])
            residualy = (Py * self[self.aperture_ind].r - ray_stop_local[0, 1])
            it += 1
        # print(f'fov{fov.item()} azimuth{azimuth.item()} success in {it}')
        # if residualx.abs() + residualy.abs() > self.APERTURE_TOLERENCE:
        #     print(f'aim ray failure at {fov.item()} {azimuth.item()}')
        #     print('------------------------------------------')
        #     print(f'residualx:{residualx.abs().item()} residualy:{residualy.abs().item()} total:{(residualx.abs()+residualy.abs()).item()}')
        #     print('------------------------------------------')
        # assert residualx + residualy < self.APERTURE_TOLERENCE, f'aim ray failure at {fov.item()} {azimuth.item()}'
        # aim ray o on the first surface's reference plane
        return Ray(o, d, wavelength=wavelength)

    @torch.no_grad()
    def aim_ref_rays(self, views, wavelengths, vig_dict):
        """
        Aiming the position of the normalized position [Px, Py] on vignetting pupil
        return the cls Ray[R5] on the first surface's reference plane
        """
        ref_ray_dict = {}
        for view in views:
            # update the field-of-view of system
            self[0].update(view)
            for wavelength in wavelengths:
                # normed pupil position
                px = torch.tensor([0., 0., 0., 1., -1.]) * (1 - vig_dict[view.item()]['VCX']) + vig_dict[view.item()]['VDX']
                py = torch.tensor([0., 1., -1., 0., 0.]) * (1 - vig_dict[view.item()]['VCY']) + vig_dict[view.item()]['VDY']
                x_init = px * self.entrance_pupil_radius
                y_init = py * self.entrance_pupil_radius
                z = torch.ones_like(x_init) * self.entrance_pupil_position
                if self[0]._type == 'infinite':
                    # infinite case: fixed angle
                    o = torch.stack([x_init, y_init, z], dim=-1)
                    d = self[0].map().unsqueeze(0).repeat(5, 1)
                elif self[0]._type == 'finite':
                    # finite case: fixed object position
                    o = torch.stack([x_init, y_init, z], dim=-1)
                    d = normalize(torch.tensor([0., 0., 0.]) - self[0].o).unsqueeze(0).repeat(5, 1)

                def get_residual(x_init, y_init):
                    o = torch.stack([x_init, y_init, z], dim=-1)
                    # form the ray
                    ray_init = Ray(o, d, wavelength=wavelength)
                    # optimize the ray's path to let it pass through the center of stop

                    ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind + 1)['ray']
                    # calcuate the initial residual
                    residual_x = (px * self[self.aperture_ind].r - ray_stop.o[..., 0])
                    residual_y = (py * self[self.aperture_ind].r - ray_stop.o[..., 1])
                    return (residual_x + residual_y).sum()

                it = 0
                # form the ray
                ray_init = Ray(o, d, wavelength=wavelength)
                ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind + 1)['ray']
                ray_stop_local = self[self.aperture_ind].to_object.transform_ray(ray_stop).o
                residual_x = (px * self[self.aperture_ind].r - ray_stop_local[..., 0])
                residual_y = (py * self[self.aperture_ind].r - ray_stop_local[..., 1])

                while (residual_x.abs() + residual_y.abs() > self.APERTURE_TOLERENCE).any() and (it < self.APERTURE_ITER_MAX):
                    """
                    测了一下还是比较线性的，梯度大致都在0.5左右，所以近似线性更新就好了
                    """
                    # 这一段需要residual.abs_()
                    # grad_x, grad_y = torch.func.grad(get_residual, argnums=(0, 1))(x_init, y_init)
                    # x_init = torch.where(grad_x != 0, x_init - residu al_x / grad_x, x_init)
                    # y_init = torch.where(grad_y != 0, y_init - residual_y / grad_y, y_init)
                    x_init += residual_x
                    y_init += residual_y

                    o = torch.stack([x_init, y_init, z], dim=-1)
                    ray_init = Ray(o, d, wavelength=wavelength)
                    ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind + 1)['ray']
                    ray_stop_local = self[self.aperture_ind].to_object.transform_ray(ray_stop).o
                    residual_x = (px * self[self.aperture_ind].r - ray_stop_local[..., 0])
                    residual_y = (py * self[self.aperture_ind].r - ray_stop_local[..., 1])
                    it += 1

                # assert (residual_x + residual_y < self.APERTURE_TOLERENCE).any(), 'aim ray failure'

                # aim ray o on the first surface's reference plane
                ref_rays = Ray(o.detach(), d, wavelength=wavelength)
                ref_rays.opd += torch.einsum('ij,ij->i', ref_rays.o - ref_rays.o[0, :], ref_rays.d)
                ref_ray_dict[view.item(), wavelength.item()] = ref_rays

        return ref_ray_dict

    # ====================================================================================
    # Propagation
    # ====================================================================================
    def propagate(self, ray, start=1, stop=None, clip=False):
        """
        stacks features during tracing
        dim: surfaces,rays,data
        """
        stacks = {k: [] for k in ('o', 'd', 't', 'valid', 'in', 'out')}
        # stacks['o'].append(ray.o)
        # stacks['d'].append(ray.d)
        # stacks['t'].append(ray.t)
        # stacks['valid'].append(ray.valid_map)
        # stacks['in'].append(ray.angle_in)
        # stacks['out'].append(ray.angle_out)
        for s in self[start:stop]:
            ray = s.propagate(ray, clip=clip)
            stacks['o'].append(ray.o)
            stacks['d'].append(ray.d)
            stacks['t'].append(ray.t)
            stacks['valid'].append(ray.valid_map)
            stacks['in'].append(ray.angle_in)
            stacks['out'].append(ray.angle_out)
        stacks['ray'] = ray
        return stacks

    # ====================================================================================
    # Paraxial Analysis
    # ====================================================================================
    @property
    def F_Number(self):
        return self.fno

    @property
    def Effective_Focal_Length(self):
        return self.effl

    @property
    def Total_Track(self):
        return self[-1].origin[2]

    @property
    def Entrance_Pupil_Position(self):
        return self.entrance_pupil_position

    @property
    def Entrance_Pupil_Diameter(self):
        return self.entrance_pupil_radius * 2

    @property
    def Exit_Pupil_Position(self):
        return self.exit_pupil_position

    @property
    def Exit_Pupil_Diameter(self):
        return self.exit_pupil_radius * 2

    @property
    def Distortion_Focal_Length(self):
        return self.dfl

    @property
    def global_exit_pupil_position(self):
        # 实际上可以根据最后一面的local2world计算参考点
        return self[-2].origin[2] + self.Exit_Pupil_Position

    @property
    def TTL(self):
        return self[-1].origin[-1]

    # ------------------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------------------
    def save_to_json(self, json_path=None):
        """
        Save the geometry of systems to the path of {json_path}
        Input Args:
            json_path: None or string, optional
                path to save the systems
        """
        if json_path is None:
            json_path = self.config
        elif isinstance(json_path, str):
            # check the existance of the directory 
            assert os.path.exists(os.path.split(json_path)[0]), \
                ValueError('save directory not exists')
        json_file = open(json_path, mode='w')

        json_content = {}
        json_content["Description"] = self.LensName.split('\n')[0]
        for surf_idx, surf in enumerate(self):
            if isinstance(surf, Conjugate):
                # object plane
                json_content["OBJECT"] = {
                    "index": int(surf_idx),
                    "roc": 0.000000000000000E+000,
                    "distance": 0.000000000000000E+000,
                    "material": "vacuum",
                    "radius": 0.000000000000000E+000,
                    "conic": 0.000000000000000E+000,
                    "ai-terms": None,
                    "variable": []
                }
            elif isinstance(surf, Surface):
                if surf_idx == self.aperture_ind:
                    json_content["STOP"] = {
                        "index": int(surf_idx),
                        "roc": 0.0 if surf.c == 0. else (1 / surf.c).item(),
                        "distance": surf.d.item(),
                        "material": surf.Material.name,
                        "radius": surf.r.item(),
                        "conic": surf.k.item(),
                        "ai-terms": None if not isinstance(surf, Aspheric) else surf.ai_terms.tolist(),
                        "variable": []
                    }
                elif surf_idx == (len(self) - 1):
                    json_content["IMAGE"] = {
                        "index": int(surf_idx),
                        "roc": 0.0 if surf.c == 0. else (1 / surf.c).item(),
                        "distance": 0.,
                        "material": surf.Material.name,
                        "radius": surf.r.item(),
                        "conic": surf.k.item(),
                        "ai-terms": None if not isinstance(surf, Aspheric) else surf.ai_terms.tolist(),
                        "variable": []
                    }
                else:
                    json_content["Standard" + str(surf_idx)] = {
                        "index": int(surf_idx),
                        "roc": 0.0 if surf.c == 0. else (1 / surf.c).item(),
                        "distance": surf.d.item(),
                        "material": surf.Material.name,
                        "radius": surf.r.item(),
                        "conic": surf.k.item(),
                        "ai-terms": None if not isinstance(surf, Aspheric) else surf.ai_terms.tolist(),
                        "variable": []
                    }

        json.dump(json_content, json_file, indent=4)

        json_file.close()


    # ====================================================================================
    # Test version methods
    # ====================================================================================

    # ====================================================================================
    # Geometric validation
    # ====================================================================================
    def geometric_val(self):
        """
        traverse the surface and check their validation in geometric
        """
        VALIDATION_SAMPLEING = 101
        valid = True
        for s_idx in range(len(self) - 1):
            if s_idx == 0:
                continue  # object plane, jump out the iteration

            if s_idx == self.aperture_ind:
                # if the surface is a plane, do not check its validation
                if (self[s_idx].c is None) or (self[s_idx].c == 0.):
                    continue  # jump to the next surface

            # compare the radius of this surface and the next surface for radius checking
            R = torch.min(torch.tensor([self[s_idx].r, self[s_idx + 1].r]))
            r_samp = torch.linspace(-R, R, VALIDATION_SAMPLEING)
            # global z-coordinates of the surface
            surf_val = self[s_idx].d + self[s_idx].surface(x=torch.zeros_like(r_samp), y=r_samp)
            surf_next_val = self[s_idx + 1].d + self[s_idx + 1].surface(x=torch.zeros_like(r_samp), y=r_samp)
            surf_diff = surf_next_val - surf_val

            if (surf_diff < 0.).any():
                valid = False

            # check the validation of surface
            if not valid:
                break

        return valid

    @torch.no_grad()
    def update_image_radius(self, view, wavelengths):
        """
        Update the radius of the image plane
        Ensuring the rays of all wavelengths in the largest fov could pass
        """
        # update the field-of-view of system
        self[0].update(view)
        R = torch.tan(self[0].angle_fov) * \
            self[1].surface(self[1].r, 0.) + \
            self[1].r

        APERTURE_SAMPLING = 201

        x_e, y_e = torch.meshgrid(torch.linspace(-R, R, APERTURE_SAMPLING),
                                  torch.linspace(-R, R, APERTURE_SAMPLING),
                                  indexing='ij')
        # generate rays and find valid map
        o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=2)
        d = self[0].map().unsqueeze(0).unsqueeze(0). \
            repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1)
        r_max = []
        for wavelength in wavelengths:
            ray = Ray(o, d, wavelength=wavelength)
            # propagate the ray to the last surface before the image plane
            valid, ray_after = self.propagate(ray, stop=-1)

            # free propagate with the valid rays
            o_after_valid = ray_after.o[valid]
            d_after_valid = ray_after.d[valid]
            t_after_valid = (self[-1].d - o_after_valid[..., 2]) / d_after_valid[..., 2]
            o_image_valid = o_after_valid + d_after_valid * t_after_valid[..., None]
            r_image_valid = torch.sqrt(torch.sum(torch.square(o_image_valid[..., 0:2]), dim=1))
            r_max.append(torch.max(r_image_valid))

        # set the max radius as the image plane
        setattr(self[-1], 'r', max(r_max))
