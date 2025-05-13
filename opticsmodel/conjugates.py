import torch
import torch.nn as nn
import numpy as np
from typing import Union
from .utils import normalize


# ==========================================================================================
# Conjugates 
# ==========================================================================================
class Conjugate(nn.Module):
    """
    The conjugate is to define the object in system
        Categories:
            Infinite plane
            Finite plane
            Object height plane (TODO)
            Image height plane (TODO)
    """
    _types = {}
    _default_type = 'infinite'
    _nickname = None
    _type = None
    _typeletter = None
    finite = None

    def __init__(self,
                 index: Union[int, torch.Tensor] = None,
                 projection='rectilinear',
                 update_radius=False):
        nn.Module.__init__(self)
        self.index = index
        self.projection = projection
        self.update_radius = update_radius

    @property
    def wideangle(self):
        # FIXME: elaborate this
        return self.projection != "rectilinear"

    def text(self):
        if self.projection != "rectilinear":
            yield "Projection: %s" % self.projection
        if self.update_radius:
            yield "Update Radius: %s" % self.update_radius
        yield "Pupil:"
        for _ in self.pupil.text():
            yield "  %s" % _

    def dict(self):
        dat = super().dict()
        dat["pupil"] = self.pupil.dict()
        if self.projection != "rectilinear":
            dat["projection"] = self.projection
        return dat

    def rescale(self, scale):
        self.pupil.rescale(scale)

    def aim(self, xy, pq, z=None, a=None):
        """
        xy 2d fractional xy object coordinate (object knows meaning)
        pq 2d fractional sagittal/meridional pupil coordinate

        aiming should be aplanatic (the grid is by solid angle
        in object space) and not paraxaxial (equal area in entrance
        beam plane)

        z pupil distance from "surface 0 apex" (also for infinite object)
        a pupil aperture (also for infinite object or telecentric pupils,
        then from z=0)

        if z, a are not provided they are takes from the (paraxial data) stored
        in object/pupil
        """
        raise NotImplementedError

    @classmethod
    def register(cls, sub):
        if sub._type is None:
            sub._type = sub.__name__.lower()
        k = cls, sub._type
        assert k not in cls._types, (k, sub, cls._types)
        cls._types[k] = sub
        return sub


@Conjugate.register
class FiniteConjugate(Conjugate):
    # NOTE: The radius of the FiniteConjucate need update after the views are addressed!
    _type = "finite"
    finite = True
    infinite = False

    def __init__(self,
                 distance: Union[float, torch.Tensor] = 0.0,
                 field_ox: Union[float, torch.Tensor] = 0.0,
                 field_oy: Union[float, torch.Tensor] = 0.0,
                 radius: Union[float, torch.Tensor] = 0.0,
                 object_radius: Union[float, torch.Tensor] = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        # the distance from origin of system to object plane
        assert distance <= 0., \
            ValueError('Finite Object plane should before the origin of system! Should be negative!')
        self.d = distance if torch.is_tensor(distance) else torch.tensor(np.asarray(distance))
        self.ox = field_ox if torch.is_tensor(field_ox) else torch.tensor(np.asarray(field_ox))
        self.oy = field_oy if torch.is_tensor(field_oy) else torch.tensor(np.asarray(field_oy))
        self.r = radius if torch.is_tensor(radius) else torch.tensor(np.asarray(radius))

        # radius of the object, if 0., object is a point
        self.object_radius = object_radius if torch.is_tensor(object_radius) else torch.tensor(np.asarray(object_radius))

    @property
    def point(self):
        # check whether the object plane is a point
        return not self.object_radius

    @property
    def angle_fov(self):
        fov_tan = torch.sqrt(self.ox ** 2 + self.oy ** 2) / self.d
        return torch.arctan(fov_tan)

    def dict(self):
        dat = super().dict()
        if self.radius:
            dat["radius"] = float(self.radius)
        return dat

    def update_distance(self, distance):
        self.d = distance if torch.is_tensor(distance) else torch.tensor(np.asarray(distance))

    def update_radius(self, radius):
        self.r = radius if torch.is_tensor(radius) else torch.tensor(np.asarray(radius))

    def update_position(self, field_ox, field_oy, object_radius=None):
        self.ox = field_ox if torch.is_tensor(field_ox) else torch.tensor(np.asarray(field_ox))
        self.oy = field_oy if torch.is_tensor(field_oy) else torch.tensor(np.asarray(field_oy))

        if object_radius:
            self.object_radius = object_radius if torch.is_tensor(object_radius) else torch.tensor(np.asarray(object_radius))

    def update(self, angle_fov, angle_azimuth=None):
        # update the ox and oy according to the new fov and azimuth
        if angle_azimuth:
            self.ox = self.d * np.tan(angle_fov) * np.sin(angle_azimuth)
            self.oy = - (self.d * np.tan(angle_fov)) * np.cos(angle_azimuth)
        else:
            self.ox = self.d * np.tan(angle_fov) * np.sin(0.)
            self.oy = - (self.d * np.tan(angle_fov))

    def map(self, origin_sample):
        """
        Get the direction from the object distance, radius, position, and the origin sample
            origin sample is the rays sampling on the origin plane of system, shape: [sample_h, sample_w, 3(xyz)]
        """
        # calculate the origin point on object
        ox_pos, oy_pos = self.r * self.ox, self.r * self.oy
        if self.object_radius:
            num_sample = np.sqrt(origin_sample.shape[0])
            ox_shift, oy_shift = torch.meshgrid(
                torch.linspace(-self.object_radius.item(), self.object_radius.item(),num_sample),
                torch.linspace(-self.object_radius.item(), self.object_radius.item(),num_sample),
                indexing='ij')
            ox, oy = ox_pos + ox_shift, oy_pos + oy_shift
        else:
            ox = torch.ones_like(origin_sample[..., 0]) * ox_pos
            oy = torch.ones_like(origin_sample[..., 1]) * oy_pos

        o = torch.stack((ox, oy, torch.ones_like(oy) * self.d), dim=2).reshape(-1, 3)
        d = normalize(origin_sample - o)  # shape: [sample_h * sample_w, 3(xyz)]
        return d

    def rescale(self, scale):
        super().rescale(scale)
        self.radius *= scale


@Conjugate.register
class InfiniteConjugate(Conjugate):
    _type = "infinite"
    finite = False
    infinite = True

    def __init__(self,
                 angle_fov: Union[float, torch.Tensor] = 0.,
                 angle_azimuth: Union[float, torch.Tensor] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.angle_fov = angle_fov * np.pi / 180.
        if angle_azimuth is None:
            self.azimuth = 0.  # direction from -y to +y direction
        else:
            # angle from the direction vector's projecton to -y axis
            self.azimuth = angle_azimuth * np.pi / 180.

    @property
    def slope(self):
        return torch.tan(self.angle_fov).item()

    def dict(self):
        dat = super().dict()
        if self.angle_fov:
            dat["angle_fov"] = float(self.angle_fov.item())
        if self.azimuth:
            dat["angle_azimuth"] = float(self.azimuth.item())
        return dat

    def update(self, angle_fov, angle_azimuth=None):
        if torch.is_tensor(angle_fov):
            self.angle_fov = angle_fov * np.pi / 180.
        else:
            self.angle_fov = torch.tensor((angle_fov * np.pi / 180.))
        if angle_azimuth is None:
            self.azimuth = torch.tensor(0.)  # direction from -y to +y direction
        else:
            # angle from the direction vector's projecton to -y axis
            self.azimuth = angle_azimuth.deg2rad()

    def text(self):
        yield "Semi-Angle: %.3g deg" % self.angle_fov
        yield from super().text()

    def map(self):
        """
        Get the direction from the fov and the azimuth
        """
        # assuming the length of direction vector is 1
        z_proj = 1 * torch.cos(self.angle_fov)
        x_proj = 1 * torch.sin(self.angle_fov) * torch.sin(self.azimuth)
        y_proj = 1 * torch.sin(self.angle_fov) * torch.cos(self.azimuth)
        # form tensor of direction (not need for normalization)
        # d = normalize(torch.tensor([x_proj, y_proj, z_proj])).to(torch.float)
        d = normalize(torch.tensor([x_proj, y_proj, z_proj]))
        return d

    def map_zemax(self):
        """
        Get the direction from the fovx(angle_fov) and fovy(azimuth)
        """
        # assuming the length of direction vector is 1

        x_proj = 1 * np.sin(self.azimuth)
        y_proj = 1 * np.sin(self.angle_fov)
        z_proj = 1 * np.sqrt(1 - x_proj ** 2 - y_proj ** 2)
        # form tensor of direction (not need for normalization)
        d = normalize(torch.tensor([x_proj, y_proj, z_proj]))
        return d
