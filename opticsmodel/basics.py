import torch
import math
import torch.nn as nn
import numpy as np


def init(dtype=torch.float64, device=torch.device('cpu')):
    print("DiffMetrology is using: {} with {}".format(device, dtype))
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)


class Ray:
    """
    Definition of a geometric ray.

    - o is the ray position
    - d is the ray direction (normalized)
    - t is the path length to the surface
    - valid_map is valid mask of rays bundle
    - wavelength is rays property
    - angle_in is the incident angle of the surface
    - angle_out is the exit angle of the surface
    - opd is the accumulated optical path
    """

    def __init__(self, o, d, wavelength, opd=None):
        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.d = torch.nn.functional.normalize(self.d, p=2, dim=-1)
        self.t = torch.zeros(o.shape[:-1])
        self.valid_map = torch.ones(o.shape[:-1]).bool()
        self.wavelength = wavelength if torch.is_tensor(wavelength) else torch.tensor(wavelength)
        self.angle_in = torch.ones(o.shape[:-1])
        self.angle_out = torch.ones(o.shape[:-1])
        self.opd = torch.zeros(o.shape[:-1]) if opd is None else opd

    def clone(self):
        ray_clone = Ray(self.o.clone(), self.d.clone(), self.wavelength.clone(), opd=self.opd.clone())
        for k in ['t', 'valid_map', 'angle_in', 'angle_out']:
            exec(f"ray_clone.{k} = self.{k}.clone()")
        return ray_clone

    def propagate2z(self, z, n=1):
        t = (z - self.o[..., 2])/self.d[..., 2]
        self.o += self.d * t[..., None]
        self.opd += t * n

# ---------------------------------------------------------------------------------------------------#
# wavelength in mm unit
fraunhofer = dict(  # http://en.wikipedia.org/wiki/Abbe_number
    i=365.01e-6,  # Hg UV
    h=404.66e-6,  # Hg violet
    g=435.84e-6,  # Hg blue
    Fp=479.99e-6,  # Cd blue
    F=486.1327e-6,  # H  blue
    e=546.07e-6,  # Hg green
    Gy=555.00e-6,  # greenish-yellow
    d=587.5618e-6,  # He yellow
    D=589.30e-6,  # Na yellow
    Cp=643.85e-6,  # Cd red
    C=656.2725e-6,  # H  red
    r=706.52e-6,  # He red
    Ap=768.20e-6,  # K  IR
    s=852.11e-6,  # Cs IR
    t=1013.98e-6,  # Hg IR
)  # unit: [mm]
# typical wavelengths
LAMBDA_F = fraunhofer["F"]
LAMBDA_d = fraunhofer["d"]
LAMBDA_C = fraunhofer["C"]
LAMBDA_Gy = fraunhofer["Gy"]
DEFAULT_WAVELENGTH = LAMBDA_d

# ---------------------------------------------------------------------------------------------------#
# sample material table
# format the material as follows:
# "material_name": {
#   "type": <type-of-material>, e.g., "abbe", "gas", "sellmeier_1", ...
#   "coeff": <coefficient-of-material>,
#   e.g., [nd, abbe number] for "abbe", [K1, L1, K2, L2, K3, L3] for "sellmeier_1", ...
# }
MATERIAL_TABLE = {
    "vacuum": {
        "type": "abbe",
        "coeff": [1., math.inf],  # [nd, abbe number]
    },
    "air": {
        "type": "gas",
        # [1.000293, math.inf]
        "coeff": [.05792105, .00167917, 238.0185, 57.362]
    },
    "mirror": {
        "type": "mirror",
        "coeff": [-1]
    },
    "occluder": [1., math.inf],
    "f2": [1.620, 36.37],
    "f15": [1.60570, 37.831],
    "uvfs": [1.458, 67.82],

    # https://shop.schott.com/advanced_optics/
    "bk10": [1.49780, 66.954],
    "n-baf10": [1.67003, 47.11],
    "n-bk7": {
        "type": "sellmeier_1",
        "coeff": [1.03961212E+000, 6.00069867E-003, 2.31792344E-001,
                  2.00179144E-002, 1.01046945E+000, 1.03560653E+002],  # [1.51680, 64.17],
    },
    "n-sf1": {
        "type": "sellmeier_1",
        "coeff": [1.608651580E+00, 1.196548790E-02, 2.377259160E-01,
                  5.905897220E-02, 1.515306530E+00, 1.355216760E+02],  # [1.71736, 29.62],
    },
    "n-sf2": [1.64769, 33.82],
    "n-sf4": [1.75513, 27.38],
    "n-sf5": [1.67271, 32.25],
    "n-sf6": [1.80518, 25.36],
    "n-sf6ht": [1.80518, 25.36],
    "n-sf8": [1.68894, 31.31],
    "n-sf10": [1.72828, 28.53],
    "n-sf11": [1.78472, 25.68],
    "sf2": [1.64769, 33.85],
    "sf4": [1.75520, 27.58],
    "sf6": [1.80518, 25.43],
    "sf18": [1.72150, 29.245],

    # HIKARI.AGF
    "baf10": [1.67, 47.05],

    # SUMITA.AGF / SCHOTT.AGF
    "sk1": [1.61030, 56.712],
    "pk1": {
        "type": "sellmeier_1",
        "coeff": [9.33773217E-01, 5.43685564E-03, 3.00776039E-01,
                  1.66532403E-02, 9.37601195E-01, 9.61682628E+01],  # [1.5038, 66.922],
    },
    "pk2": {
        "type": "sellmeier_1",
        "coeff": [1.20921993E+00, 7.33975047E-03, 6.68873331E-02,
                  2.86495917E-02, 1.01818125E+00, 1.01852918E+02],  # [1.5182, 65.054],
    },
    "fk5": {
        "type": "sellmeier_1",
        "coeff": [8.44309338E-01, 4.75111955E-03, 3.44147824E-01,
                  1.49814849E-02, 9.10790213E-01, 9.78600293E+01],  # [1.4875, 70.406],
    },
    "lak10": {
        "type": "sellmeier_1",
        "coeff": [1.67482975E+00, 8.49324758E-03, 2.22668828E-01,
                  3.30463141E-02, 1.13956536E+00, 8.04372553E+01],  # [1.72, 50.41],
    },
    "laf2": {
        "type": "sellmeier_1",
        "coeff": [1.75302282E+00, 9.57243296E-03, 2.13978997E-01,
                  4.02787459E-02, 1.09939444E+00, 1.02533259E+02],  # [1.744, 44.72],
    },
    "balf5": {
        "type": "sellmeier_1",
        "coeff": [1.23751018E+00, 8.08113000E-03, 1.16757584E-01,
                  3.49868372E-02, 8.53086255E-01, 1.10314185E+02],  # [1.5414, 53.629],
    },
    "sk14": {
        "type": "sellmeier_1",
        "coeff": [9.36155374E-01, 4.61716525E-03, 5.94052018E-01,
                  1.68859270E-02, 1.04374583E+00, 1.03736265E+02],  # [1.6031, 60.597],
    },
    "sf1": {
        "type": "sellmeier_1",
        "coeff": [1.55912923E+00, 1.21481001E-02, 2.84246288E-01,
                  5.34549042E-02, 9.68842926E-01, 1.12174809E+02],  # [1.7174, 29.513],
    },
    "sf5": {
        "type": "sellmeier_1",
        "coeff": [1.46141885E+00, 1.11826126E-02, 2.47713019E-01,
                  5.08594669E-02, 9.49995832E-01, 1.12041888E+02],  # [1.6727, 32.21],
    },
    "sf8": {
        "type": "sellmeier_1",
        "coeff": [1.49514446E+00, 1.14990201E-02, 2.62529687E-01,
                  5.17170156E-02, 9.69567597E-01, 1.13458641E+02],  # [1.6889, 31.176],
    },
    "sf54": {
        "type": "sellmeier_1",
        "coeff": [1.60371792E+00, 1.24263783E-02, 3.10740554E-01,
                  5.51896382E-02, 1.03053604E+00, 1.15762058E+02],  # [1.7408, 28.091],
    },
    "lakn6": {
        "type": "sellmeier_1",
        "coeff": [1.15477318E+00, 5.83257597E-03, 4.97049217E-01,
                  1.94613961E-02, 1.00553588E+00, 9.78667698E+01],  # [1.6425, 57.962],
    },
    "lakn7": {
        "type": "sellmeier_1",
        "coeff": [1.23679889E+00, 6.10105538E-03, 4.45051837E-01,
                  2.01388334E-02, 1.01745888E+00, 9.06380380E+01],  # [1.6516, 58.518],
    },
    "lakn12": {
        "type": "sellmeier_1",
        "coeff": [1.17365704E+00, 5.77031797E-03, 5.88992398E-01,
                  2.00401678E-02, 9.78014394E-01, 9.54873482E+01],  # [1.6779, 55.2],
    },
    "sk2": {
        "type": "sellmeier_1",
        "coeff": [1.281890120E+00, 7.271916400E-03, 2.577382580E-01,
                  2.428235270E-02, 9.681860400E-01, 1.103777730E+02],  # [1.6074, 56.65],
    },
    "sk4": {
        "type": "sellmeier_1",
        "coeff": [1.39388834E+00, 7.69147355E-03, 1.64510721E-01,
                  2.71947227E-02, 9.63522479E-01, 9.92757639E+01],  # [1.6127, 58.63],
    },
    "sk16": {
        "type": "sellmeier_1",
        "coeff": [1.343177740E+00, 7.046873390E-03, 2.411443990E-01,
                  2.290050000E-02, 9.943179690E-01, 9.275085260E+01],  # [1.62040, 60.306],
    },
    "kf9": {
        "type": "sellmeier_1",
        "coeff": [1.16421867E+00, 7.87954228E-03, 1.18007401E-01,
                  3.72520495E-02, 9.23054885E-01, 1.09924834E+02],  # [1.5234, 51.493],
    },
    "ssk4a": {
        "type": "sellmeier_1",
        "coeff": [1.38336169E+00, 7.87154985E-03, 1.87125107E-01,
                  2.89352221E-02, 9.44170902E-01, 1.05742424E+02],  # [1.6176, 55.142],
    },
    "sf56a": {
        "type": "sellmeier_1",
        "coeff": [1.70579259E+00, 1.33874699E-02, 3.44223052E-01,
                  5.79561608E-02, 1.09601828E+00, 1.21616024E+02],  # [1.7847, 26.077],
    },
    "f1": {
        "type": "sellmeier_1",
        "coeff": [1.36192123E+00, 1.02049225E-02, 2.09162831E-01,
                  4.80290444E-02, 9.07276681E-01, 1.09211919E+02],  # [1.6259, 35.7],
    },
    "f5": {
        "type": "sellmeier_1",
        "coeff": [1.310446300E+00, 9.586330480E-03, 1.960342600E-01,
                  4.576276270E-02, 9.661297700E-01, 1.150118830E+02],  # [1.6034, 38.03],
    },

    # OHARA.AGF
    "fd11": {
        "type": "schott",
        "coeff": [3.08005200E+00, -2.35111600E-02, 1.94608800E-02,
                  9.65106600E-03, -1.24477700E-03, 8.77739000E-05],  # [1.7847, 25.704],
    },
    "s-lah51": {
        "type": "sellmeier_1",
        "coeff": [1.82586991E+00, 9.35297152E-03, 2.83023349E-01,
                  3.73803057E-02, 1.35964319E+00, 1.00655798E+02],  # [1.7859, 44.203],
    },
    "s-lah52": {
        "type": "sellmeier_1",
        "coeff": [1.85390925E+00, 9.55320687E-03, 2.97925555E-01,
                  3.93816850E-02, 1.39382086E+00, 1.02706848E+02],  # [1.7995, 42.225],
    },
    "s-lah53": {
        "type": "sellmeier_1",
        "coeff": [1.91811619E+00, 1.02147684E-02, 2.53724399E-01,
                  4.33176011E-02, 1.39473885E+00, 1.01938021E+02],  # [1.8061, 40.926],
    },


    # https://www.pgo-online.com/intl/B270.html
    "b270": [1.52290, 58.50],

    # https://refractiveindex.info, nD at 589.3 nm
    "s-nph1": [1.8078, 22.76],
    "d-k59": [1.5175, 63.50],

    "flint": [1.6200, 36.37],
    "pmma": [1.491756, 58.00],
    "polycarb": [1.585470, 30.00],

    # honor20
    "mc-pcd4-40": {
        "type": "schott",
        "coeff": [2.58314710E+000, -9.75023440E-003, 1.36153740E-002,
                  3.63461220E-004, -2.11870820E-005, 1.15361320E-006],  # [1.6192, 63.855],
    },
    "ep9000": {
        "type": "schott",
        "coeff": [2.67158942E+000, -9.88033522E-003, 2.31490098E-002,
                  9.46210022E-003, -1.30260155E-003, 1.19691096E-004],  # [1.6707, 19.238],
    },
    "apl5014cl": {
        "type": "schott",
        "coeff": [2.39240344E+000, -3.17013191E-002, -1.76719919E-002,
                  9.49949989E-003, -1.27481919E-003, 6.65182214E-005],  # [1.5439, 55.951],
    },
    "k26r": {
        "type": "schott",
        "coeff": [2.39341385E+000, -5.52710285E-002, -3.05566524E-002,
                  1.20870398E-002, -1.51685332E-003, 7.48343683E-005],  # [1.5348, 55.664],
    },
    "apl5014cl_apel": {
        "type": "schott",
        "coeff": [2.34232979E+000, -7.05971129E-004, 1.44801133E-002,
                  1.72130051E-004, -2.42252894E-006, 9.97640535E-007],  # [1.5445, 55.987],
    },
    "polystyr": {
        "type": "schott",
        "coeff": [2.44598368E+000, 2.21428933E-005, 2.72988569E-002,
                  3.01210852E-004, 8.88934888E-005, -1.75707929E-006],  # [1.5905, 30.867],
    },
    "h-k9l": {
        "type": "sellmeier_1",
        "coeff": [6.14555251E-001, 1.45987884E-002, 6.56775017E-001,
                  2.87769588E-003, 1.02699346E+000, 1.07653051E+002],  # [1.5168, 64.199],
    }
}

class Material(nn.Module):
    """
    Optical materials for computing the refractive indices.

    support several categories of material

    1. first is the abbe material, where the formulation can easily expressed as:

    n(\lambda) = A + (\lambda - \lambda_ref) / (\lambda_long - \lambda_short) * (1 - A) / B

    where the two constants A and B can be computed from nD (index at 589.3 nm) and V (abbe number).

    2. second is the coeff material, where the formulation is defined by different equations:

    schott / sellmeier_1 / sellmeier_squared / sellmeier_squared_transposed / conrady / ...

    for more details of the coeff material, please refer to the following.

    """

    def __init__(self, name: str = None):
        # initilize with the nn.Module cls
        nn.Module.__init__(self)
        # TODO: register optimized parameters
        self.name = 'vacuum' if name is None else name.lower()

        self.material = MATERIAL_TABLE.get(self.name)
        if not torch.is_tensor(self.material['coeff']):
            self.material['coeff'] = torch.tensor(self.material['coeff'])

        self.nd = self.refractive_index(LAMBDA_d)

    # ==========================================================================
    # return the refractive index and abbe number
    # ==========================================================================

    def refractive_index(self, wavelength):
        n_fn = getattr(self, "n_%s" % self.material['type'])
        n = n_fn(wavelength * 1e3, self.material['coeff'])
        if 'mirror' in self.material.keys():
            n = -n if self.material['mirror'] else n

        return n

    def abbe_number(self):
        return torch.tensor(np.asarray((self.refractive_index(LAMBDA_d) - 1) / (self.refractive_index(LAMBDA_F) - self.refractive_index(LAMBDA_C))))

    # ==========================================================================
    # calculating refractive index with different dispersion equation
    # ==========================================================================

    def n_abbe(self, w, c):
        return c[0] + (w - LAMBDA_d) / (LAMBDA_C - LAMBDA_F) * (1 - c[0]) / c[1]

    def n_mirror(self, w, c):
        return torch.tensor(-1)

    def n_schott(self, w, c):
        n = c[0] + c[1] * w ** 2
        for i, ci in enumerate(c[2:]):
            n += ci * w ** (-2 * (i + 1))
        return torch.sqrt(n)

    def n_sellmeier(self, w, c):
        w2 = w ** 2
        c0, c1 = c.reshape(-1, 2).T
        return torch.sqrt(1. + (c0 * w2 / (w2 - c1 ** 2)).sum())

    def n_sellmeier_1(self, w, c):
        w2 = w ** 2
        c0, c1 = c.reshape(-1, 2).T
        return torch.sqrt((1. + (c0 * w2 / (w2 - c1)).sum()))

    def n_sellmeier_squared_transposed(self, w, c):
        w2 = w ** 2
        c0, c1 = c.reshape(2, -1)
        return torch.sqrt(1. + (c0 * w2 / (w2 - c1)).sum())

    def n_conrady(self, w, c):
        return c[0] + c[1] / w + c[2] / w ** 3.5

    def n_herzberger(self, w, c):
        l = 1. / (w ** 2 - .028)
        return c[0] + c[1] * l + c[2] * l ** 2 + c[3] * w ** 2 + c[4] * w ** 4 + c[5] * w ** 6

    def n_sellmeier_offset(self, w, c):
        w2 = w ** 2
        c0, c1 = c[1:1 + (c.shape[0] - 1) // 2 * 2].reshape(-1, 2).T
        return torch.sqrt(1. + c[0] + (c0 * w2 / (w2 - c1 ** 2)).sum())

    def n_sellmeier_squared_offset(self, w, c):
        w2 = w ** 2
        c0, c1 = c[1:1 + (c.shape[0] - 1) // 2 * 2].reshape(-1, 2).T
        return torch.sqrt(1. + c[0] + (c0 * w2 / (w2 - c1)).sum())

    def n_handbook_of_optics1(self, w, c):
        return torch.sqrt(c[0] + (c[1] / (w ** 2 - c[2])) - (c[3] * w ** 2))

    def n_handbook_of_optics2(self, w, c):
        return torch.sqrt(c[0] + (c[1] * w ** 2 / (w ** 2 - c[2])) - (c[3] * w ** 2))

    def n_extended2(self, w, c):
        n = c[0] + c[1] * w ** 2 + c[6] * w ** 4 + c[7] * w ** 6
        for i, ci in enumerate(c[2:6]):
            n += ci * w ** (-2 * (i + 1))
        return torch.sqrt(n)

    def n_hikari(self, w, c):
        n = c[0] + c[1] * w ** 2 + c[2] * w ** 4
        for i, ci in enumerate(c[3:]):
            n += ci * w ** (-2 * (i + 1))
        return torch.sqrt(n)

    def n_gas(self, w, c):
        c0, c1 = c.reshape(2, -1)
        return 1. + (c0 / (c1 - w ** -2)).sum()

    def n_gas_offset(self, w, c):
        return c[0] + self.n_gas(w, c[1:])

    def n_refractiveindex_info(self, w, c):
        c0, c1 = c[9:].reshape(-1, 2).T
        return torch.sqrt(c[0] + c[1] * w ** c[2] / (w ** 2 - c[3] ** c[4]) +
                          c[5] * w ** c[6] / (w ** 2 - c[7] ** c[8]) + (c0 * w ** c1).sum())

    def n_retro(self, w, c):
        w2 = w ** 2
        a = c[0] + c[1] * w2 / (w2 - c[2]) + c[3] * w2
        return torch.sqrt(2 + 1 / (a - 1))

    def n_cauchy(self, w, c):
        c0, c1 = c[1:].reshape(-1, 2).T
        return c[0] + (c0 * w ** c1).sum()

    def n_polynomial(self, w, c):
        return torch.sqrt(self.n_cauchy(w, c))

    def n_exotic(self, w, c):
        return torch.sqrt(c[0] + c[1] / (w ** 2 - c[2]) +
                          c[3] * (w - c[4]) / ((w - c[4]) ** 2 + c[5]))

    # ==========================================================================
    # output API for checking
    # ==========================================================================

    def dict(self):
        dat = {}
        if self.name:
            dat['name'] = self.name
        if self.material['type']:
            dat['type'] = self.material['type']
        if self.material['coeff'] is not None:
            dat['coeff'] = self.material['coeff']
        if 'mirror' in self.material.keys():
            dat['mirror'] = self.material['mirror']

        return dat

    def __str__(self):
        material_info = self.dict()
        output_base = material_info['name'].upper() + '\n' + \
                      f'    Material type: ' + material_info['type'] + '\n' + \
                      f'    Material coeff: ' + \
                      np.array_str(np.asarray(material_info['coeff'].cpu())) + '\n'
        if 'mirror' in material_info.keys():
            if material_info['mirror']:
                output_mirror = f'    Material MIRROR: True' + '\n'
        else:
            output_mirror = f'    Material MIRROR: False' + '\n'
        return output_base + output_mirror