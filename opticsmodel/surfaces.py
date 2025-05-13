import torch.distributed.optim
from typing import Union
from .utils import *
from .basics import Material, DEFAULT_WAVELENGTH


class Surface(nn.Module):
    # ======================================================================================
    # Initialization （初始化）
    # ======================================================================================
    def __init__(self,
                 index: Union[int, torch.Tensor] = None,
                 radius: Union[float, torch.Tensor] = 0.0,
                 distance: Union[float, torch.Tensor] = 0.0,
                 distance_prev: Union[float, torch.Tensor] = 0.,
                 distance_after: Union[float, torch.Tensor] = 0.,
                 material: str = None,
                 material_prev: str = None,
                 origin: Union[list, np.array, torch.Tensor] = [],
                 shift: Union[list, np.array, torch.Tensor] = np.zeros(3),
                 theta_xyz: Union[list, np.array, torch.Tensor] = np.zeros(3),
                 variables: Union[list, str] = []
                 ):

        # basic information
        self.index = index

        # surface information
        self.r = torch.tensor(radius)
        self.d = torch.tensor(distance_after)
        self.d_prev = torch.tensor(distance_prev)  # depreciated!
        self.clip = False
        # global axial distance
        distance = torch.tensor(distance)
        # initilize with the nn.Module cls, for nn.Parameters, Material, and TransformMixin statement
        nn.Module.__init__(self)
        #
        self.Material = Material(material)
        self.Material_Prev = Material(material_prev)

        if len(origin) == 0:
            self.origin = torch.stack((torch.zeros_like(distance), torch.zeros_like(distance), distance), dim=0)
        else:
            self.origin = origin if torch.is_tensor(origin) else torch.tensor(origin)

        self.shift = shift if torch.is_tensor(shift) else torch.tensor(shift)
        self.theta_xyz = theta_xyz if torch.is_tensor(theta_xyz) else torch.tensor(theta_xyz)

        # register the parameters according to the variable
        self.variables = variables
        if self.variables.__class__.__name__ == 'list':
            for var in self.variables:
                if var in dir(self):
                    exec('self.{x}.requires_grad_(True)'.format(x=var))
                    # exec('self.{x} = nn.Parameter(self.{x})'.format(x=var))
                    # self.register_parameter(str(self.index) + var, getattr(self, var))
                    # exec(f'self.register_parameter(\'{str(self.index)+var}\', self.{var})') # also work
        elif self.variables.__class__.__name__ == 'str':
            if self.variables in dir(self):
                exec('self.{x}.requires_grad_(True)'.format(x=self.variables))
                # exec('self.{x} = nn.Parameter(self.{x})'.format(x=self.variables))
                # self.register_parameter(str(self.index) + self.variables, getattr(self, self.variables))
                # exec('self.register_parameter(str(self.index) + {x}, self.{x})'.format(x=self.variables))


        # generate gradient graph
        self._compute_transformation_quaternions()

        # There are the parameters controlling the accuracy of ray tracing.
        self.NEWTONS_MAXITER = 20
        self.NEWTONS_TOLERANCE_TIGHT = 1e-10  # in [mm], i.e. 1e-4 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 1e-9  # in [mm], i.e. 1e-3 [nm] here (up to <10 [nm])
        self.SURFACE_SAMPLING = 31
        self.NEWTONS_STEP_BOUND = 1.  # in [mm], maximum iteration step in Newton's iteration

    # ======================================================================================
    # Common methods (must not be overridden in child class)
    # ======================================================================================
    def _compute_transformation_quaternions(self):
        """
        we compute to_world transformation given the input positional parameters (angles)a
            alpha = torch.deg2rad(self.theta_x)
            beta = torch.deg2rad(self.theta_y)
            gamma = torch.deg2rad(self.theta_z)
        """
        z = torch.tensor(0.0)

        q_x = torch.stack([torch.cos(torch.deg2rad(self.theta_xyz[0]) / 2), torch.sin(torch.deg2rad(self.theta_xyz[0]) / 2), z, z])
        q_y = torch.stack([torch.cos(torch.deg2rad(self.theta_xyz[1]) / 2), z, torch.sin(torch.deg2rad(self.theta_xyz[1]) / 2), z])
        q_z = torch.stack([torch.cos(torch.deg2rad(self.theta_xyz[2]) / 2), z, z, torch.sin(torch.deg2rad(self.theta_xyz[2]) / 2)])

        q_xyz = quaternion_raw_multiply(quaternion_raw_multiply(q_x, q_y), q_z)

        self.to_object = Transformation_quaternions(q_xyz, self.shift, self.origin)
        self.to_world = self.to_object.inverse()
        # self.to_world = Transformation_quaternions(q_xyz.detach() * torch.tensor([1, -1, -1, -1]), self.shift.detach(), self.origin, flag=0)

    def update_transform(self, origin):
        """
        update_transform and update gradient map for updated d
        """
        self.origin = origin
        self._compute_transformation_quaternions()

    def refractive_index(self, wavelength):
        """
        Return the refractive index of the material after this surface, i.e., n1
        """
        return self.Material.refractive_index(wavelength)

    def refractive_index_prev(self, wavelength):
        """
        Return the refractive index of the material before this surface, i.e., n0
        """
        return self.Material_Prev.refractive_index(wavelength)

    @property
    def abbe_number(self):
        """
        Return the abbe number of the material after this surface, i.e., vd
        """
        return self.Material.abbe_number()

    @property
    def abbe_number_prev(self):
        """
        Return the abbe number of the material before this surface, i.e., vd_
        """
        return self.Material_Prev.abbe_number()

    def surface_with_offset(self, x, y):
        """
        Return the global z coordinates of the point on surface
        """
        return self.surface(x, y) + self.origin[2]

    def aperture_valid(self, xy):
        sdf_approx = length2(xy) - self.r ** 2
        return (sdf_approx <= 1E-8).bool()

    def paraxial_matrix(self, wavelength=DEFAULT_WAVELENGTH):
        """
        2x2 block matrix, M = [[A, B], [C, D]]
        Refraction at a curved interface [[1, 0 ], [c*(n1/n2-1), n1/n2]]
        Propagation in free space or in a medium of constant refractive index [[1, d], [0, 1]]
        abcd = R @ P
        """
        D = self.refractive_index_prev(wavelength)/self.refractive_index(wavelength)
        C = self.c * (D - 1)
        A = 1 + C * self.d
        B = D * self.d
        abcd = torch.stack((A, B, C, D)).reshape(2, 2)
        return abcd

    # ====================================================================================== 
    # Key methods for propagation 
    # ======================================================================================
    def propagate(self, ray, clip=False):
        """
        ray reactions
        """
        ray = ray.clone()
        # local coor according to the origin of this surface
        ray_obj = self.to_object.transform_ray(ray)

        ray_obj = self.ray_surface_intersection(ray_obj)
        ray_obj = self.refract(ray_obj)

        if clip | self.clip:
            valid_a = self.aperture_valid(ray_obj.o[..., :2])
            ray_obj.valid_map *= valid_a

        # to world coordinates
        ray_world = self.to_world.transform_ray(ray_obj)

        return ray_world

    def refract(self, ray, gv=0):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://en.wikipedia.org/wiki/Snell%27s_law
        gv: grating vector
        """
        n1 = self.refractive_index_prev(ray.wavelength)
        n2 = self.refractive_index(ray.wavelength)
        if self.Material.name.lower() == 'mirror':
            mu = - (ray.d + gv)
        else:
            mu = (n1 * ray.d + gv) / n2
        # normal with nan
        normal = - self.surface_normal(ray.o[..., 0], ray.o[..., 1])
        incident_angle = torch.arccos(torch.einsum('...k,...k', normal, ray.d)).rad2deg()
        cosi = torch.einsum('...k,...k', normal, mu)
        cost2 = 1. - length2(mu) + cosi ** 2
        valid_d = (cost2 > 0.) & ray.valid_map
        sr = torch.sqrt(torch.clamp(cost2, min=1e-16))
        # sr = torch.sqrt(cost2)
        # sr = torch.sqrt(1 - length2(mu) * torch.einsum('...k,...k', normal, mu) ** 2 * valid_d) # must check
        # here we do not have to do normalization because if both wi and n are normalized,
        # then output is also normalized.
        d_tmp = mu + normal * (sr - cosi)[..., None]
        d_tmp[~valid_d] = ray.d[~valid_d]
        exit_angle = torch.arccos((torch.einsum('...k,...k', normal, d_tmp)).abs()).rad2deg()
        ray.d = d_tmp
        ray.angle_in = incident_angle
        ray.angle_out = exit_angle
        ray.valid_map *= valid_d
        return ray

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, oz):
        """
        Returns g(x,y)+h(z) and dot((g'x,g'y,h'), (dx,dy,dz)).
        (Note: this default implementation is not efficient)
        """
        x = ox + t * dx
        y = oy + t * dy
        z = oz + t * dz
        s = self.surface(x, y) - z
        sx, sy, sz = self.surface_derivatives(x, y)
        return s, sx * dx + sy * dy + sz * dz

    def surface_normal(self, x, y):
        """
        The normal of surface in the position (x, y), i.e., [NORX, NORY, NORZ] in Zemax
        Output format: [NORX, NORY, NORZ] (normalized to 1)
        """
        return normalize(torch.stack(self.surface_derivatives(x, y), dim=-1))

    @torch.no_grad()
    def find_t0_in_aperture(self, ray):
        # ray must in local coordinate
        x, y = torch.meshgrid(torch.linspace(-self.r, self.r, self.SURFACE_SAMPLING),
                              torch.linspace(-self.r, self.r, self.SURFACE_SAMPLING), indexing='ij')
        valid_map = self.aperture_valid(torch.stack((x, y), dim=-1))
        xyz = torch.stack((x[valid_map], y[valid_map], self.surface(x[valid_map], y[valid_map])), dim=-1).reshape(-1, 3)
        rho = xyz[:, None, :] - ray.o[None, ...]
        distance = length(torch.linalg.cross(torch.broadcast_to(ray.d, rho.shape), rho, dim=-1) + 1e-8)
        nearest_point = xyz[torch.argmin(distance, dim=0)]
        t0 = torch.einsum('ij,ij->i', nearest_point-ray.o, ray.d)
        return t0

    # === Virtual methods (must be overridden)
    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        """
        raise NotImplementedError()

    def ray_surface_intersection(self, ray):
        raise NotImplementedError

    def surface_derivatives(self, x, y):
        """
        Returns \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        (Note: this default implementation is not efficient)
        """
        raise NotImplementedError()


class Spheric(Surface):
    """
    Spheric surface
    """

    def __init__(self,
                 roc: Union[float, torch.Tensor] = 0.0,
                 conic: Union[float, torch.Tensor] = 0.0,
                 **kwargs):
        self.maximum_term = 0
        if roc is not None and roc != 0.:
            self.c = 1 / roc if torch.is_tensor(roc) else torch.tensor(1 / roc)
        elif roc == 0.:
            self.c = torch.tensor(0.)
        if conic is not None:
            self.k = conic if torch.is_tensor(conic) else torch.tensor(conic)
        else:
            self.k = torch.tensor(0.)
        super().__init__(**kwargs)

    def ray_surface_intersection(self, ray):
        """
        solve the quadric equation to obtain the intersection of ray and spheric
        active: valid map for rays bundle
        """
        if self.c == 0:  # degrade to plane propagation
            t = (0. - ray.o[..., 2]) / ray.d[..., 2]
            o_tmp = ray.o + t.unsqueeze(-1) * ray.d
            valid_s = ray.valid_map
        else:
            if self.k is None:
                do = (ray.o * ray.d).sum(-1)
                dd = 1.
                oo = torch.square(ray.o).sum(-1)
            else:
                k = torch.tensor([1., 1., 1 + self.k])
                do = (ray.o * ray.d * k).sum(-1)
                dd = (torch.square(ray.d) * k).sum(-1)
                oo = (torch.square(ray.o) * k).sum(-1)

            d = self.c * do - ray.d[..., -1]
            e = self.c * dd
            f = self.c * oo - 2 * ray.o[..., -1]
            valid_s = (torch.square(d) - e * f > 0) & ray.valid_map
            g = torch.sqrt(torch.clamp(torch.square(d) - e * f, min=1e-16))
            t = -(d + g) / e
            o_tmp = ray.o + t[..., None] * ray.d

        o_tmp[~valid_s] = ray.o[~valid_s]
        ray.o = o_tmp
        ray.valid_map *= valid_s
        ray.t = t
        ray.opd += self.refractive_index_prev(ray.wavelength) * t
        return ray

    # ========================
    def surface(self, x, y):
        r2 = x ** 2 + y ** 2
        tmp = r2 * self.c
        sq = torch.clamp((1 - (1 + self.k) * tmp * self.c), min=1e-16)
        total_surface = tmp / (1 + torch.sqrt(sq))
        return total_surface

    # === Private methods
    def surface_derivatives(self, x, y):
        r2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.k) * self.c ** 2 * r2
        tmp = torch.sqrt(torch.clamp(1. - alpha_r2, min=1E-8))
        total_derivative = self.c * (1. + tmp - 0.5 * alpha_r2) / (tmp * (1. + tmp) ** 2)
        dsdx, dsdy = total_derivative * 2 * x, total_derivative * 2 * y
        return dsdx, dsdy, -torch.ones_like(dsdx)

    def surface_valid(self, x, y):
        valid = ((1 + self.k) * self.c**2 * (x**2 + y**2) < 1)
        return valid

class Aspheric(Spheric):
    """
    Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.
    NOTE: high order aspheric term start from r^{4}, while the extended aspheric start from r^{2}.
    The ai_2 in zemax could be coverd by c, so we don't consider this value as high order aspheric term
    """

    def __init__(self,
                 ai: Union[list, np.array, torch.Tensor] = [0.0],
                 **kwargs):
        super().__init__(**kwargs)
        self.maximum_term = len(ai)
        ai = torch.tensor(np.array(ai))
        for order in range(self.maximum_term):
            exec('self.ai_{x} = ai[{y}]'.format(x=str(2 * (order + 1) + 2), y=order))

        # register the parameters according to the variable
        if self.variables.__class__.__name__ == 'list':
            if 'ai' in self.variables:
                for order in range(self.maximum_term):
                    var = 'ai_' + str(2 * (order + 1) + 2)
                    exec('self.ai_{x}.requires_grad_(True)'.format(x=str(2 * (order + 1) + 2)))
                    # exec('self.ai_{x} = nn.Parameter(self.ai_{x})'.format(x=str(2 * (order + 1) + 2)))
                    # exec('self.register_parameter(str(self.index) + var, getattr(self, var))')
                    # exec(f'self.register_parameter(\'{str(self.index)+var}\', self.{var})') # also work
        elif self.variables.__class__.__name__ == 'str':
            if self.variables == 'ai':
                for order in range(self.maximum_term):
                    var = 'ai_' + str(2 * (order + 1) + 2)
                    exec('self.{x}.requires_grad_(True)'.format(x=self.variables))
                    # exec('self.{x} = nn.Parameter(self.{x})'.format(x=self.variables))
                    # exec('self.register_parameter(str(self.index) + var, getattr(self, var))')
                    # exec('self.register_parameter(str(self.index) + {x}, self.{x})'.format(x=self.variables))


    # === Common methods
    @property
    def ai_terms(self):
        terms = []
        for order in range(self.maximum_term):
            exec('terms.append(self.ai_{x})'.format(x=str(2 * (order + 1) + 2)))
        terms = torch.tensor(terms)
        return terms

    def surface(self, x, y):
        r2 = x ** 2 + y ** 2
        sph_surface = super().surface(x, y)
        higher_surface = 0
        # 从r4开始的
        if self.maximum_term:
            for inv_order in np.flip(range(self.maximum_term)):
                higher_surface = r2 * higher_surface + getattr(self, 'ai_' + str(2 * (inv_order + 1) + 2))
            higher_surface = higher_surface * r2 ** 2
        total_surface = sph_surface + higher_surface
        return total_surface

    def ray_surface_intersection(self, ray):
        """
        Get the intersections of ray and surface
        Returns:
        - p: intersection point
        - g: explicit funciton
        """
        # using newtons method
        def residual(o_tmp):
            # check valid need?
            return self.surface(o_tmp[..., 0], o_tmp[..., 1]) - o_tmp[..., 2]
        def func_derivarte(o_tmp, ray_d):
            dxdt, dydt, dzdt = ray_d[..., 0], ray_d[..., 1], ray_d[..., 2]
            dfdx, dfdy, dfdz = self.surface_derivatives(o_tmp[..., 0], o_tmp[..., 1])
            dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
            return dfdt

        t0 = self.find_t0_in_aperture(ray)
        # impl Newton ref: dO: A differentiable engine for deep lens design of computational imaging systems.
        with torch.no_grad():
            it = 0
            t = t0
            res = 1E8 * torch.ones_like(ray.t)
            while(torch.abs(res[ray.valid_map]) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
                it += 1
                o_tmp = ray.o + ray.d * t.unsqueeze(-1)
                # check valid
                # valid = self.surface_valid(o_tmp[..., 0], o_tmp[..., 0]) & ray.valid_map
                # ft = self.sag(new_x, new_y, valid) + self.d - new_o[..., 2]
                res = residual(o_tmp)
                derivate = func_derivarte(o_tmp, ray.d)
                t = t - torch.clamp(res/derivate, -self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)
                # res[res.isnan()] = 0
            t1 = t - t0
        t = t0 + t1
        o_tmp = ray.o + ray.d * t.unsqueeze(-1)
        res = residual(o_tmp)
        derivate = func_derivarte(o_tmp, ray.d)
        t = t - torch.clamp(res / derivate, -self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)
        o_tmp = ray.o + ray.d * t.unsqueeze(-1)
        valid_s = torch.abs(res) < self.NEWTONS_TOLERANCE_LOOSE
        # note that the p is only the valid intersection
        o_tmp[~valid_s] = ray.o[~valid_s]
        ray.o = o_tmp
        ray.valid_map *= valid_s
        ray.t = t
        ray.opd += self.refractive_index_prev(ray.wavelength) * t
        return ray

    def surface_derivatives(self, x, y):
        r2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.k) * self.c ** 2 * r2
        tmp = torch.sqrt(torch.clamp(1. - alpha_r2, min=1e-8))
        sph_derivative = self.c * (1. + tmp - 0.5 * alpha_r2) / (tmp * (1. + tmp) ** 2)
        higher_derivative = 0
        if self.maximum_term:
            for inv_order in np.flip(range(self.maximum_term)):
                higher_derivative = r2 * higher_derivative + \
                                    (inv_order + 2) * getattr(self, 'ai_' + str(2 * (inv_order + 1) + 2))
        total_derivative = sph_derivative + higher_derivative * r2
        dsdx, dsdy = total_derivative * 2 * x, total_derivative * 2 * y
        return dsdx, dsdy, -torch.ones_like(dsdx)


