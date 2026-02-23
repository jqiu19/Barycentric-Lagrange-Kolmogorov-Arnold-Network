import equinox as eqx
import jax
from jax import random
from jax.nn import gelu, silu, tanh
import jax.numpy as jnp
import numpy as np
from utils import split_kanshape


def get_network(args, input_dim, output_dim, interval, normalizer, keys):
    if args.network == 'mlp':
        model = MLP(input_dim=input_dim, output_dim=output_dim, N_features=args.features, N_layers=args.layers,
                    normalizer=normalizer, activation=args.activation,
                    key=keys[0])
    elif args.network == 'modifiedmlp':
        model = modifiedMLP(input_dim=input_dim, output_dim=output_dim, N_features=args.features, N_layers=args.layers,
                            normalizer=normalizer, activation=args.activation,
                            key=keys[0])
    elif args.network == 'kan':
        features = split_kanshape(input_dim, output_dim, args.kanshape)
        model = KAN(features=features, interval=interval, degree=args.degree, normalizer=normalizer, key=keys[0])
    elif args.network == 'sinckan':
        features = split_kanshape(input_dim, output_dim, args.kanshape)
        model = sincKAN(features=features, degree=args.degree, len_h=args.len_h, normalizer=normalizer,
                        init_h=args.init_h, decay=args.decay, skip=args.skip, activation=args.activation, key=keys[0])
    elif args.network == 'chebykan':
        features = split_kanshape(input_dim, output_dim, args.kanshape)
        model = chebyKAN(features=features, degree=args.degree, normalizer=normalizer, key=keys[0])
    elif args.network == 'barykan':
        features = split_kanshape(input_dim, output_dim, args.kanshape)
        model = baryKAN(
            features=features,
            degree=args.degree,
            interval=interval,
            normalizer=normalizer,
            key=keys[0],
        )
    else:
        assert False, f'{args.network} does not exist'
    return model


class MLP(eqx.Module):
    matrices: list
    biases: list
    activation: jax.nn
    normalizer: list

    def __init__(self, input_dim, output_dim, N_features, N_layers, normalizer, key, activation='tanh'):
        keys = random.split(key, N_layers + 1)
        features = [input_dim, ] + [N_features, ] * (N_layers - 1) + [output_dim, ]
        self.matrices = [random.normal(key, (f_in, f_out)) / jnp.sqrt((f_in + f_out) / 2) for f_in, f_out, key in
                         zip(features[:-1], features[1:], keys)]
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out in zip(features[:-1], features[1:])]
        if activation == 'silu':
            self.activation = silu
        elif activation == 'gelu':
            self.activation = gelu
        else:
            self.activation = tanh
        self.normalizer = [normalizer]

    def __call__(self, input, frozen_para):

        f = input @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            # f = tanh(f)
            f = self.activation(f)
            f = f @ self.matrices[i] + self.biases[i]
        return f

    def get_frozen_para(self):
        frozen = []
        return frozen


class modifiedMLP(eqx.Module):
    matrices: list
    biases: list
    matrices_modified: list
    biases_modified: list
    activation: jax.nn
    normalizer: list

    def __init__(self, input_dim, output_dim, N_features, N_layers, normalizer, key, activation='tanh'):
        keys = random.split(key, N_layers + 1)
        features = [input_dim, ] + [N_features, ] * (N_layers - 1) + [output_dim, ]
        self.matrices = [random.normal(key, (f_in, f_out)) / jnp.sqrt((f_in + f_out) / 2) for f_in, f_out, key in
                         zip(features[:-1], features[1:], keys)]
        self.matrices_modified = [
            random.normal(key, (features[0], features[1])) / jnp.sqrt((features[0] + features[1]) / 2),
            random.normal(key, (features[0], features[1])) / jnp.sqrt((features[0] + features[1]) / 2)]

        self.biases = [jnp.zeros((f_out,)) for f_in, f_out in zip(features[:-1], features[1:])]
        self.biases_modified = [jnp.zeros((features[1],)), jnp.zeros((features[1],))]

        if activation == 'silu':
            self.activation = silu
        elif activation == 'gelu':
            self.activation = gelu
        else:
            self.activation = tanh

        self.normalizer = [normalizer]

    def __call__(self, input, frozen_para):
        input = self.normalizer[0](input)
        u = input @ self.matrices_modified[0] + self.biases_modified[0]
        v = input @ self.matrices_modified[1] + self.biases_modified[1]

        u = self.activation(u)
        v = self.activation(v)

        f = input @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            # f = tanh(f)
            f = self.activation(f)
            f = f * u + (1 - f) * v
            f = f @ self.matrices[i] + self.biases[i]
        return f

    def get_frozen_para(self):
        frozen = []
        return frozen


class KAN(eqx.Module):
    layers: list
    activation: jax.nn
    normalizer: list

    def __init__(self, features, interval, normalizer, key, degree=10, activation='tanh'):
        keys = random.split(key, len(features) + 1)
        self.layers = [KANLayers(f_in, f_out, degree, interval, key) for f_in, f_out, key in
                       zip(features[:-1], features[1:], keys)]
        if activation == 'tanh':
            self.activation = tanh
        self.normalizer = [normalizer]

    def __call__(self, x, frozen_para):
        for i in range(len(self.layers)):
            # f = tanh(f)
            x = self.layers[i](x, frozen_para[i])
        return x

    def get_frozen_para(self):
        frozen = []
        for i in range(len(self.layers)):
            frozen.append(self.layers[i].get_frozen_para())
        return frozen


class chebyKAN(eqx.Module):
    layers: list
    normalizer: list
    activation: jax.nn

    def __init__(self, features, normalizer, key, degree=10, activation='tanh'):
        keys = random.split(key, len(features) + 1)
        self.layers = [ChebyLayers(f_in, f_out, degree, key) for f_in, f_out, key in
                       zip(features[:-1], features[1:], keys)]
        self.normalizer = [normalizer]
        if activation == 'tanh':
            self.activation = tanh

    def __call__(self, x, frozen_para):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.layers[i](x, frozen_para[i])
        return x

    def get_frozen_para(self):
        frozen = []
        for i in range(len(self.layers)):
            frozen.append(self.layers[i].get_frozen_para())
        return frozen


class sincKAN(eqx.Module):
    layers: list
    normalizer: list

    def __init__(self, features, normalizer, key, degree, len_h, decay, activation, init_h, skip):
        keys = random.split(key, len(features) + 1)
        self.layers = [SincLayers(f_in, f_out, degree, key, len_h=len_h, init_h=init_h, activation=activation,
                                  decay=decay, skip=skip) for f_in, f_out, key in
                       zip(features[:-1], features[1:], keys)]
        self.normalizer = [normalizer]

    def __call__(self, x, frozen_para):
        x = self.normalizer[0](x)
        for i in range(len(self.layers)):
            x = self.layers[i](x, frozen_para[i])
        return x

    def get_frozen_para(self):
        frozen = []
        for i in range(len(self.layers)):
            frozen.append(self.layers[i].get_frozen_para())
        return frozen

    def update_basis(self, frozen_para, add_num, key):
        assert add_num >= 2, print('adding degree should be greater than 2')
        keys = random.split(key, 2 * len((self.layers)))
        for i in range(len((self.layers))):
            degree = self.layers[i].degree
            new_k = jnp.arange(-jnp.floor((degree + add_num) / 2), jnp.ceil((degree + add_num) / 2) + 1)
            new_k = jnp.expand_dims(new_k, axis=(0, 1))
            old_k = frozen_para[i]['k']
            N_left = old_k[0, 0, 0] - new_k[0, 0, 0]
            N_right = -old_k[0, 0, -1] + new_k[0, 0, -1]
            input_dim, output_dim, len_h = self.layers[i].coeffs.shape[:3]
            coef_left = jnp.zeros((input_dim, output_dim, len_h, int(N_left)))
            coef_right = jnp.zeros((input_dim, output_dim, len_h, int(N_right)))
            self.layers[i] = eqx.tree_at(lambda t: t.degree, self.layers[i], degree + add_num)
            frozen_para[i]['k'] = new_k
            self.layers[i] = eqx.tree_at(lambda t: t.coeffs, self.layers[i],
                                         jnp.concatenate([coef_left, self.layers[i].coeffs, coef_right], -1))

class IntervalNormalizer:
    def __init__(self, low, high, enable=True):
        self.low = float(low)
        self.high = float(high)
        self.enable = bool(enable)

    def __call__(self, x):
        if not self.enable:
            return x
        # Map [low, high] -> [-1, 1]
        return 2.0 * (x - self.low) / (self.high - self.low) - 1.0


def chebyshev_second_kind_nodes(degree: int) -> jnp.ndarray:
    """
    x_j = cos(j*pi/N), j=0..N, where N=degree (so degree+1 points)
    """
    N = int(degree)
    j = jnp.arange(0, N + 1, dtype=jnp.float32)
    return jnp.cos(j * jnp.pi / N)


def barycentric_weights_cheb2_closed_form(x_support_1d: jnp.ndarray) -> jnp.ndarray:
    """
    EXACTLY matches your rational_baryrat.py closed-form implementation for Chebyshev 2nd kind nodes:
        w_j = (-1)^j * delta_j, delta endpoints = 1/2
    """
    n = x_support_1d.size - 1
    w = jnp.empty((n + 1,), dtype=x_support_1d.dtype)
    # alternate signs
    w = w.at[0::2].set(1.0)
    w = w.at[1::2].set(-1.0)
    # endpoint scaling
    if n > 0:
        w = w.at[0].set(w[0] * 0.5)
        w = w.at[n].set(w[n] * 0.5)
    else:
        w = w.at[0].set(w[0] * 0.5)
    return w


class baryKAN(eqx.Module):
    layers: list
    normalizer: list
    input_interval_norm: IntervalNormalizer

    def __init__(self, features, degree, interval, normalizer, key):
        keys = random.split(key, len(features) + 1)
        self.layers = [
            BaryLayers(f_in, f_out, degree=degree, key=k)
            for f_in, f_out, k in zip(features[:-1], features[1:], keys)
        ]
        self.normalizer = [normalizer]
        self.input_interval_norm = IntervalNormalizer(interval[0], interval[1], enable=True)

    def __call__(self, x, frozen_para):
        # First normalize raw input x into [-1, 1] to match x_support domain
        x = self.normalizer[0](x)
        x = self.input_interval_norm(x)

        L = len(self.layers)
        for i in range(L):
            x = self.layers[i](x, frozen_para[i])

            # Requirement: outputs from layer 0..n-2 must be in [-1,1]
            # Use tanh (guaranteed) for all hidden layers except last.
            if i < L - 1:
                x = jnp.tanh(x)
        return x

    def get_frozen_para(self):
        frozen = []
        for i in range(len(self.layers)):
            frozen.append(self.layers[i].get_frozen_para())
        return frozen


class BaryLayers(eqx.Module):
    """
    One Barycentric Lagrange KAN layer:
      - Each edge (input_dim i -> output_dim o) has its own theta[i,o,:] (independent init)
      - x_support and w are shared and fixed (frozen_para)
      - y_o = mean_i phi_{i->o}(x_i) * scale[i,o] + tanh(x) @ linear[i,o]   (residual like KAN)
    """
    degree: int
    input_dim: int
    output_dim: int

    # learnable
    coeffs_phi_raw: jnp.ndarray     # theta: (in_dim, out_dim, degree+1)
    scale: jnp.ndarray              # (in_dim, out_dim)
    linear: jnp.ndarray             # (in_dim, out_dim)
    bias: jnp.ndarray               # (out_dim,)

    def __init__(self, input_dim, output_dim, degree, key):
        self.degree = int(degree)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        k1, k2, k3, k4 = random.split(key, 4)

        # Independent theta init per edge (like torch.rand * 0.1)
        self.coeffs_phi_raw = random.uniform(
            k1,
            (self.input_dim, self.output_dim, self.degree + 1),
            minval=0.0,
            maxval=0.1,
        ).astype(jnp.float32)

        # Similar to KAN: scale and linear residual mixing
        self.scale = jnp.ones((self.input_dim, self.output_dim), dtype=jnp.float32)
        # Xavier-ish init for linear
        self.linear = random.normal(k2, (self.input_dim, self.output_dim), dtype=jnp.float32) / jnp.sqrt(
            (self.input_dim + self.output_dim) / 2.0
        )
        self.bias = jnp.zeros((self.output_dim,), dtype=jnp.float32)

    def __call__(self, x, frozen_para):
        """
        x: (input_dim,)
        returns: (output_dim,)
        """
        # Evaluate barycentric rational for each edge (i,o) at scalar x_i.
        # x_support, w: shape (N,) where N=degree+1
        x_support = frozen_para["x_support"]  # (N,)
        w = frozen_para["w"]                  # (N,)
        theta = self.coeffs_phi_raw           # (in_dim, out_dim, N)

        # diff: (in_dim, N)
        diff = x[:, None] - x_support[None, :]
        exact = diff == 0.0

        diff_safe = jnp.where(exact, 1.0, diff)
        inv = 1.0 / diff_safe  # (in_dim, N)

        # num: (in_dim, out_dim)
        num = jnp.sum((w[None, None, :] * theta) * inv[:, None, :], axis=-1)

        # den: (in_dim,)
        den = jnp.sum(w[None, :] * inv, axis=-1)

        out = num / den[:, None]  # (in_dim, out_dim)

        # exact-hit handling: if x_i equals a support node x_j, output theta[i,o,j]
        any_exact = jnp.any(exact, axis=-1)  # (in_dim,)
        j_idx = jnp.argmax(exact.astype(jnp.int32), axis=-1)  # (in_dim,)
        theta_pick = jnp.take_along_axis(theta, j_idx[:, None, None], axis=-1).squeeze(-1)  # (in_dim,out_dim)
        out = jnp.where(any_exact[:, None], theta_pick, out)

        # KAN-style aggregation: mean over input dims + residual
        y_interp = jnp.mean(out * self.scale, axis=0)  # (out_dim,)
        y_res = jnp.tanh(x) @ self.linear + self.bias  # (out_dim,)
        y = y_interp
        return y

    def get_frozen_para(self):
        x_support = chebyshev_second_kind_nodes(self.degree)  # (degree+1,)
        w = barycentric_weights_cheb2_closed_form(x_support)  # (degree+1,)
        return {"x_support": x_support, "w": w}



class KANLayers(eqx.Module):
    k: int
    dx: float
    G: int
    input_dim: int
    activation: jax.nn
    coeffs: list
    interval: list

    def __init__(self, input_dim, output_dim, degree, interval, key, activation='silu'):
        self.k = 3
        self.G = degree - self.k
        self.coeffs = [random.normal(key, (input_dim, output_dim, degree)) / jnp.sqrt(input_dim * (degree + 1)),
                       jnp.ones((input_dim, output_dim)), jnp.ones((input_dim, output_dim))]

        self.dx = (interval[1] - interval[0]) / self.G
        self.interval = interval
        self.input_dim = input_dim
        # (in_dim*out_dim,G+2k+1)

        if activation == 'silu':
            self.activation = silu
        elif activation == 'gelu':
            self.activation = gelu
        else:
            self.activation = tanh

    def __call__(self, x, frozen_para):
        basis = self.get_spline_basis(x, frozen_para)
        spl = jnp.einsum("id,iod->io", basis, self.coeffs[0])
        res = self.activation(x)

        y = jnp.mean(spl * self.coeffs[1], axis=0) + res @ self.coeffs[2]

        return y

    def get_spline_basis(self, x, frozen_para):
        grid = frozen_para['grid']
        x = jnp.expand_dims(x, axis=1)
        basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
        for k in range(1, self.k + 1):
            left_term = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right_term = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])

            basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

        return basis_splines

    def get_frozen_para(self):
        grid = jnp.tile(jnp.expand_dims(
            jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * self.dx + self.interval[0], 0),
            (self.input_dim, 1))
        return {'grid': grid}


class ChebyLayers(eqx.Module):
    degree: int
    coeffs: jnp.array

    def __init__(self, input_dim, output_dim, degree, key):
        self.degree = degree
        self.coeffs = random.normal(key, (input_dim, output_dim, degree + 1)) / jnp.sqrt(input_dim * (degree + 1))

    def __call__(self, x, frozen_para):
        x = jnp.tile(jnp.expand_dims(x, axis=1), (1, self.degree + 1))
        x = jnp.arccos(x)

        k = frozen_para['k']
        x = x * k
        x = jnp.cos(x)
        y = jnp.einsum("id,iod->o", x, self.coeffs)
        return y

    def get_frozen_para(self):
        k = jnp.arange(0, self.degree + 1, 1).astype(float)
        return {'k': k}


class SincLayers(eqx.Module):
    degree: int
    len_h: int
    init_h: float
    coeffs: jnp.array
    decay: str
    beta: jnp.array
    alpha: jnp.array
    activation: jax.nn
    skip: bool

    def __init__(self, input_dim, output_dim, degree, key, init_h, len_h, activation, decay,
                 skip, initialization='Xavier'):
        self.degree = degree
        self.decay = decay
        if initialization == 'Xavier':
            self.coeffs = random.normal(key, (input_dim, output_dim, len_h, (degree + 1))) / jnp.sqrt(
                input_dim * (degree + 1))
        else:
            self.coeffs = random.normal(key, (input_dim, output_dim, len_h, (degree + 1)))

        self.len_h = len_h
        self.init_h = init_h

        self.skip = skip
        self.alpha = jnp.ones((input_dim, output_dim))
        self.beta = jnp.zeros((output_dim,))
        if activation == 'tanh':
            self.activation = tanh
        if activation == 'sin':
            self.activation = jnp.sin
        if activation == 'cos':
            self.activation = jnp.cos
        else:
            self.activation = None

    def __call__(self, x, frozen_para):

        if self.skip:
            y_eqt = x @ self.alpha + self.beta

        if self.activation is not None:
            x = self.activation(x)
        x = jnp.tile(jnp.expand_dims(x, axis=(1, 2)), (1, 1, self.degree + 1))

        k = frozen_para['k']
        h = frozen_para['h']
        x = x / h + k

        x_interp = jnp.sinc(x)
        #x_interp = jnp.sin(x)/(x + 1e-6)

        y = jnp.einsum("ikd,iokd->o", x_interp, self.coeffs)
        if self.skip:
            y = y_eqt + y

        return y

    def get_frozen_para(self):
        k = jnp.arange(-jnp.floor(self.degree / 2), jnp.ceil(self.degree / 2) + 1)
        k = jnp.expand_dims(k, axis=(0, 1))

        if self.decay == 'inverse':
            h = 1 / (self.init_h * (1 + jnp.arange(self.len_h)))
        elif self.decay == 'exp':
            h = 1 / (self.init_h ** (1 + jnp.arange(self.len_h)))
        else:
            assert False, f'{self.decay} does not exist'

        h = jnp.expand_dims(h, axis=(0, 2))

        return {'k': k, 'h': h}
