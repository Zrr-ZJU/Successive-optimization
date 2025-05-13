import numpy as np
import torch
import torch.nn as nn


class CoherentPsfOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, d, grid, opd, k):
        ctx.save_for_backward(o, d, grid, opd, k)
        r = grid[:, :, None, :] - o[None, None, :, :]
        dr = torch.einsum('...k,...k->...', d[None, None, :, :], r)
        amp = torch.einsum('...k->...', torch.exp(1j * k * (opd + dr))*d[None, None, :, 2])
        psf = torch.abs(amp) ** 2
        return psf

    @staticmethod
    def backward(ctx, grad_output):
        o, d, grid, opd, k = ctx.saved_tensors
        grad_o = torch.empty_like(o) if ctx.needs_input_grad[0] else None
        grad_d = torch.empty_like(d) if ctx.needs_input_grad[1] else None
        grad_grid = torch.empty_like(grid) if ctx.needs_input_grad[2] else None
        grad_opd = torch.empty_like(opd) if ctx.needs_input_grad[3] else None
        # dr = torch.einsum('...k,...k->...', d[None, None, :, :], grid[:, :, None, :] - o[None, None, :, :])
        phase = k * (opd + torch.einsum('...k,...k->...', d[None, None, :, :], grid[:, :, None, :] - o[None, None, :, :]))
        base_elec = torch.polar(phase.new_ones(phase.shape), phase)

        # shape [x,y,ray]
        dI_dEReal = 2 * torch.sum((base_elec * d[None, None, :, 2]).real, dim=-1, keepdim=True)
        dI_dEImag = 2 * torch.sum((base_elec * d[None, None, :, 2]).imag, dim=-1, keepdim=True)
        # dEReal_dPhase = -base_elec.imag * d[None, None, :, 2]
        # dEImag_dPhase = base_elec.real * d[None, None, :, 2]
        # dEReal_dDz = base_elec.real
        # dEImag_dDz = base_elec.imag
        # dI_dPhase = dI_dEReal * (-base_elec.imag * d[None, None, :, 2]) + dI_dEImag * base_elec.real * d[None, None, :, 2]
        # dI_dDz = dI_dEReal * base_elec.real + dI_dEImag * base_elec.imag
        grad_phase = (dI_dEReal * (-base_elec.imag * d[None, None, :, 2]) + dI_dEImag * base_elec.real * d[None, None, :, 2]) * grad_output[..., None]
        grad_Dz = torch.sum((dI_dEReal * base_elec.real + dI_dEImag * base_elec.imag) * grad_output[..., None], dim=[0,1])[:, None]
        grad_D = torch.cat([grad_Dz.new_zeros(grad_Dz.shape), grad_Dz.new_zeros(grad_Dz.shape), grad_Dz], dim=-1)
        # grad_dr = k * grad_phase
        # dr = (grid-o)*d
        # 注意shape要对应
        if ctx.needs_input_grad[0]:
            # (grid-o)d，所以这里是-1, 另外None是为了扩充data维度，因为是点积，权重为1
            grad_o = torch.sum((k * grad_phase)[..., None] * (-d[None, None, :, :]), dim=[0, 1])
        if ctx.needs_input_grad[1]:
            grad_d = torch.sum((k * grad_phase)[..., None] * (grid[:, :, None, :] - o[None, None, :, :]), dim=[0, 1]) + grad_D
        if ctx.needs_input_grad[2]:
            grad_grid = torch.sum((k * grad_phase)[..., None] * d[None, None, :, :], dim=[2])
        if ctx.needs_input_grad[3]:
            grad_opd = k * torch.sum(grad_phase, dim=[0, 1])

        return grad_o, grad_d, grad_grid, grad_opd, None


class Transformation_quaternions(nn.Module):
    """
    Rigid Transformation.
    - R is the rotation matrix.
    - t is the translational vector.
    - flag=0 means inverse transform
    """
    def __init__(self, R, t, o, flag=1):
        nn.Module.__init__(self)
        self.R = R if torch.is_tensor(R) else torch.tensor(np.asarray(R))
        self.t = t if torch.is_tensor(t) else torch.tensor(np.asarray(t))
        self.o = o if torch.is_tensor(o) else torch.tensor(np.asarray(o))
        self.flag = flag

    def transform_point(self, o):
        if self.flag == 1:
            o = o - self.o - self.t
            real_parts = o.new_zeros(o.shape[:-1] + (1,))
            o = torch.cat((real_parts, o), -1)
            return quaternion_raw_multiply(quaternion_raw_multiply(self.R * torch.tensor([1, -1, -1, -1]), o), self.R)[..., 1:]
        else:
            real_parts = o.new_zeros(o.shape[:-1] + (1,))
            o = torch.cat((real_parts, o), -1)
            out_o = quaternion_raw_multiply(quaternion_raw_multiply(self.R * torch.tensor([1, -1, -1, -1]), o), self.R)[..., 1:]
            out_o = out_o + self.o + self.t
            return out_o

    def transform_vector(self, d):
        real_parts = d.new_zeros(d.shape[:-1] + (1,))
        d = torch.cat((real_parts, d), -1)
        return quaternion_raw_multiply(quaternion_raw_multiply(self.R * torch.tensor([1, -1, -1, -1]), d), self.R)[..., 1:]

    def transform_ray(self, ray):
        ray.o = self.transform_point(ray.o)
        ray.d = self.transform_vector(ray.d)
        return ray

    def inverse(self):
        RT = self.R * torch.tensor([1, -1, -1, -1])
        t = self.t
        o = self.o
        return Transformation_quaternions(RT, t, o, flag=0)

    def update_o(self, o):
        self.o = o if torch.is_tensor(o) else torch.tensor(np.asarray(o))


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    # print(a.shape)
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def length2(d):
    return torch.sum(d ** 2, dim=-1)


def length(d):
    return torch.sqrt(length2(d))


def normalize(d):
    return d / length(d)[..., None]


