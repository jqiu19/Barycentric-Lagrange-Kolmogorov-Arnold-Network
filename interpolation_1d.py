#!/usr/bin/env python3
# interpolation_1d.py

import os
import sys
import time
import argparse

# -------------------------
# Parse args early (so we can set CUDA_VISIBLE_DEVICES before importing JAX)
# -------------------------
parser = argparse.ArgumentParser(description="BarylagrangeKAN")

parser.add_argument("--datatype", type=str, default="bl", help="type of data")
parser.add_argument("--npoints", type=int, default=500, help="the number of total dataset")
parser.add_argument("--ntest", type=int, default=1000, help="the number of testing dataset")
parser.add_argument("--ntrain", type=int, default=500, help="the number of training dataset for each epochs")
parser.add_argument("--ite", type=int, default=20, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=50000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (fallback/legacy)")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--noise", type=int, default=0, help="add noise or not, 0: no noise, 1: add noise")

parser.add_argument(
    "--normalization",
    type=int,
    default=0,
    help="add normalization or not, 0: no normalization, 1: add normalization",
)

parser.add_argument("--interval", type=str, default="0.0,1.0", help="boundary of the interval")
parser.add_argument("--network", type=str, default="mlp", help="type of network")
parser.add_argument("--kanshape", type=str, default="16", help="shape of the network (KAN)")
parser.add_argument("--degree", type=int, default=100, help="degree of polynomials")
parser.add_argument("--features", type=int, default=100, help="width of the network")
parser.add_argument("--layers", type=int, default=10, help="depth of the network")
parser.add_argument("--len_h", type=int, default=2, help="length of k for sinckan")
parser.add_argument("--embed_feature", type=int, default=10, help="embedding features of the modified MLP")
parser.add_argument("--device", type=int, default=7, help="cuda number (index inside CUDA_VISIBLE_DEVICES)")
parser.add_argument("--init_h", type=int, default=2, help="initial value of h")
parser.add_argument("--decay", type=str, default="inverse", help="decay type")
parser.add_argument("--skip", type=bool, default=False, help="skip connection")
parser.add_argument("--activation", type=str, default="tanh", help="activation function")

# wandb args
parser.add_argument("--wandb_project", type=str, default="BarylagrangeKAN", help="wandb project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/team (optional)")
parser.add_argument("--wandb_name", type=str, default=None, help="run name (optional)")
parser.add_argument("--wandb_tags", type=str, default="", help="comma-separated tags")
parser.add_argument("--wandb_log_interval", type=int, default=1, help="log every N steps")

args = parser.parse_args()

# -------------------------
# Environment setup (important: do this BEFORE importing jax)
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

# wandb defaults (offline by default; won't override if you exported WANDB_MODE=online)
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

# Your repo layout
sys.path.append("../")

# -------------------------
# Imports after env setup
# -------------------------
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, grad, vmap
from jax import debug
from jax.tree_util import GetAttrKey, DictKey

import equinox as eqx
import optax
import wandb

from data import get_data
from networks import get_network
from utils import normalization


def cosine_anneal_lr(lr_init: float, eta_min: float, T_max: int):
    """CosineAnnealingLR equivalent:
       eta_min + 0.5*(lr_init-eta_min)*(1+cos(pi*t/T_max))
    """
    T_max = max(int(T_max), 1)

    def schedule(step):
        step = jnp.minimum(step, T_max)
        cos = 0.5 * (1.0 + jnp.cos(jnp.pi * step / T_max))
        return eta_min + (lr_init - eta_min) * cos

    return schedule


def build_phi_mask(params):
    """
    Return pytree of booleans with same structure as params:
    True iff that leaf corresponds to `coeffs_phi_raw` by path inspection.

    Notes:
    - `params = eqx.filter(model, eqx.is_array)` gives a pytree matching model,
      with non-array leaves as None. We must handle None safely.
    """
    flat_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    mask_flat = []
    for path, leaf in flat_with_path:
        is_phi = False
        for k in path:
            if isinstance(k, GetAttrKey) and k.name == "coeffs_phi_raw":
                is_phi = True
                break
            if isinstance(k, DictKey) and k.key == "coeffs_phi_raw":
                is_phi = True
                break
        # Non-array leaves are None in params; keep mask False there
        mask_flat.append(bool(is_phi) if leaf is not None else False)
    return jax.tree_util.tree_unflatten(treedef, mask_flat)


def net(model, x, frozen_para):
    # model expects shape [in_dim], then returns shape [out_dim]
    return model(jnp.stack([x]), frozen_para)[0]


def compute_loss(model, ob_xy, frozen_para):
    # ob_xy: [B, 2], columns: x, y
    output = vmap(net, (None, 0, None))(model, ob_xy[:, 0], frozen_para)
    return 100.0 * jnp.mean((output - ob_xy[:, 1]) ** 2)


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


def _zero_if_not_match(g, m, want_phi: bool):
    # g can be None (non-array leaves). Must keep None.
    if g is None:
        return None
    if want_phi:
        return g if m else jnp.zeros_like(g)
    else:
        return jnp.zeros_like(g) if m else g


def _add_updates(u1, u2):
    if u1 is None and u2 is None:
        return None
    if u1 is None:
        return u2
    if u2 is None:
        return u1
    return u1 + u2


@eqx.filter_jit
def make_step(model, ob_xy, frozen_para, tx_other, tx_phi, opt_state_other, opt_state_phi, phi_mask):
    loss, grads = compute_loss_and_grads(model, ob_xy, frozen_para)
    grads = eqx.filter(grads, eqx.is_array)  # keep same structure, non-arrays -> None

    # debug grad norm (global across array leaves)
    #debug.print("grad_norm: {}", optax.global_norm(grads))

    # split grads by phi_mask
    grads_other = jax.tree_util.tree_map(lambda g, m: _zero_if_not_match(g, m, want_phi=False), grads, phi_mask)
    grads_phi = jax.tree_util.tree_map(lambda g, m: _zero_if_not_match(g, m, want_phi=True), grads, phi_mask)

    params = eqx.filter(model, eqx.is_array)

    updates_other, opt_state_other = tx_other.update(grads_other, opt_state_other, params)
    updates_phi, opt_state_phi = tx_phi.update(grads_phi, opt_state_phi, params)

    updates = jax.tree_util.tree_map(_add_updates, updates_other, updates_phi)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state_other, opt_state_phi


def train(key):
    # Generate sample data
    lowb, upb = map(float, args.interval.split(","))
    interval = [lowb, upb]

    x_train = np.linspace(lowb, upb, num=args.npoints)[:, None]
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]

    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train)
    y_target = y_train.copy()

    if args.noise == 1:
        sigma = 0.1
        y_train = y_train + np.random.normal(0, sigma, y_train.shape)

    y_test = generate_data(x_test)
    normalizer = normalization(x_train, args.normalization)

    # IMPORTANT: use jnp array to avoid host↔device issues
    ob_xy = jnp.array(np.concatenate([x_train, y_train], axis=-1))

    input_dim = 1
    output_dim = 1

    # Build model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()

    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f"total parameters: {param_count}")
    wandb.summary["total_parameters"] = int(param_count)

    # Training hyperparameters
    N_train = int(args.ntrain)
    N_epochs = int(args.epochs)
    ite = int(args.ite)
    total_steps = int(ite * N_epochs)

    # Clamp batch size for random.choice(replace=False)
    N_train = max(1, min(N_train, int(ob_xy.shape[0])))

    # Params and phi mask
    params = eqx.filter(model, eqx.is_array)
    phi_mask = build_phi_mask(params)

    # Schedules
    lr_other = cosine_anneal_lr(1e-3, 1e-6, total_steps)
    lr_phi = cosine_anneal_lr(3e-3, 1e-6, total_steps)

    weight_decay = 1e-5
    tx_other = optax.adamw(learning_rate=lr_other, weight_decay=weight_decay)
    tx_phi = optax.adamw(learning_rate=lr_phi, weight_decay=weight_decay)

    opt_state_other = tx_other.init(params)
    opt_state_phi = tx_phi.init(params)

    # Sample initial batch
    keys = random.split(keys[-1], 2)
    input_points = random.choice(keys[0], ob_xy, shape=(N_train,), replace=False)

    history = []
    T = []

    for j in range(total_steps):
        t1 = time.time()
        loss, model, opt_state_other, opt_state_phi = make_step(
            model, input_points, frozen_para, tx_other, tx_phi, opt_state_other, opt_state_phi, phi_mask
        )
        t2 = time.time()

        step_time = t2 - t1
        T.append(step_time)
        history.append(float(loss))


        lr_now_other = float(lr_other(j))
        lr_now_phi = float(lr_phi(j))

        # logging
        if (j % max(1, int(args.wandb_log_interval))) == 0:
            wandb.log(
                {
                    "train/loss": float(loss),
                    "train/lr_other": lr_now_other,
                    "train/lr_phi": lr_now_phi,
                    "perf/step_time_sec": float(step_time),
                    "perf/steps_per_sec": float(1.0 / step_time) if step_time > 0 else 0.0,
                },
                step=j,
            )

        # your original evaluation cadence (every N_epochs steps)
        if j % N_epochs == 0:
            keys = random.split(keys[-1], 2)
            input_points = random.choice(keys[0], ob_xy, shape=(N_train,), replace=False)

            train_y_pred = vmap(net, (None, 0, None))(model, jnp.array(x_train)[:, 0], frozen_para)
            train_mse_error = jnp.mean((train_y_pred.flatten() - jnp.array(y_target).flatten()) ** 2)
            train_relative_error = (
                jnp.linalg.norm(train_y_pred.flatten() - jnp.array(y_target).flatten())
                / jnp.linalg.norm(jnp.array(y_target).flatten())
            )

            print(f"ite:{j},mse:{train_mse_error:.2e},relative:{train_relative_error:.2e}")

            wandb.log(
                {
                    "train/mse": float(train_mse_error),
                    "train/relative": float(train_relative_error),
                },
                step=j,
            )

    # Final eval (train/test)
    avg_time = float(np.mean(np.array(T))) if len(T) else 0.0
    if avg_time > 0:
        print(f"time: {1 / avg_time:.2e}ite/s")

    train_y_pred = vmap(net, (None, 0, None))(model, jnp.array(x_train)[:, 0], frozen_para)
    train_mse_error = jnp.mean((train_y_pred.flatten() - jnp.array(y_target).flatten()) ** 2)
    train_relative_error = (
        jnp.linalg.norm(train_y_pred.flatten() - jnp.array(y_target).flatten())
        / jnp.linalg.norm(jnp.array(y_target).flatten())
    )
    print(f"training mse: {train_mse_error:.2e},relative: {train_relative_error:.2e}")

    y_pred = vmap(net, (None, 0, None))(model, jnp.array(x_test)[:, 0], frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - jnp.array(y_test).flatten()) ** 2)
    relative_error = (
        jnp.linalg.norm(y_pred.flatten() - jnp.array(y_test).flatten())
        / jnp.linalg.norm(jnp.array(y_test).flatten())
    )
    print(f"testing mse: {mse_error:.2e},relative: {relative_error:.2e}")

    # wandb summary
    wandb.summary["train_mse"] = float(train_mse_error)
    wandb.summary["train_relative"] = float(train_relative_error)
    wandb.summary["test_mse"] = float(mse_error)
    wandb.summary["test_relative"] = float(relative_error)
    wandb.summary["avg_step_time_sec"] = float(avg_time)
    wandb.summary["steps_per_sec"] = float(1.0 / avg_time) if avg_time > 0 else 0.0
    wandb.summary["total_steps"] = int(total_steps)

    # Save model and results
    model_path = f"{args.datatype}_{args.network}_{args.seed}.eqx"
    eqx.tree_serialise_leaves(model_path, model)

    npz_path = f"{args.datatype}_{args.network}_{args.seed}.npz"
    np.savez(
        npz_path,
        loss=np.array(history),
        avg_time=avg_time,
        y_pred=np.array(y_pred),
        y_test=np.array(y_test),
        y_coarse_pred=np.array(train_y_pred),
        y_coarse_test=np.array(y_target),
    )

    try:
        wandb.save(model_path)
        wandb.save(npz_path)
    except Exception as e:
        print("wandb.save failed:", e)

    # Write CSV results (keep original)
    header = "datatype, network, seed, final_loss_mean, training_time, total_ite, mse, relative, fine_mse, fine_relative"
    save_here = "results_SUPPLEM20250428.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = (
        f"\n{args.datatype},{args.network},{args.seed},{history[-1]},{np.sum(np.array(T))},"
        f"{total_steps},{float(train_mse_error)},{float(train_relative_error)},{float(mse_error)},{float(relative_error)}"
    )
    with open(save_here, "a") as f:
        f.write(res)

    return model_path


def eval(key):
    lowb, upb = map(float, args.interval.split(","))
    interval = [lowb, upb]

    x_train = np.linspace(lowb, upb, num=args.npoints)[:, None]
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]

    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train)
    y_target = y_train.copy()

    if args.noise == 1:
        sigma = 0.1
        y_train = y_train + np.random.normal(0, sigma, y_train.shape)

    y_test = generate_data(x_test)
    normalizer = normalization(x_train, args.normalization)

    input_dim = 1
    output_dim = 1

    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)

    model_path = f"{args.datatype}_{args.network}_{args.seed}.eqx"
    frozen_para = model.get_frozen_para()
    model = eqx.tree_deserialise_leaves(model_path, model)

    # Special logging for sinckan (keep original behavior)
    if args.network == "sinckan":
        netlayer = lambda layer, x, fp: layer(jnp.stack([x]), fp)
        z0 = vmap(netlayer, (None, 0, None))(model.layers[0], jnp.array(x_train)[:, 0], frozen_para[0])
        z1 = vmap(netlayer, (None, 0, None))(model.layers[1], jnp.array(x_train)[:, 0], frozen_para[1])
        np.savez("inter.npz", z0=np.array(z0), z1=np.array(z1))
        try:
            wandb.save("inter.npz")
        except Exception:
            pass

    y_pred = vmap(net, (None, 0, None))(model, jnp.array(x_test)[:, 0], frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - jnp.array(y_test).flatten()) ** 2)
    relative_error = (
        jnp.linalg.norm(y_pred.flatten() - jnp.array(y_test).flatten())
        / jnp.linalg.norm(jnp.array(y_test).flatten())
    )
    print(f"mse: {mse_error},relative: {relative_error}")

    wandb.log({"eval/mse": float(mse_error), "eval/relative": float(relative_error)})

    # Plot fit
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test, "r", label="Original Data")
    plt.plot(x_test, np.array(y_pred).reshape(-1, 1), "b-", label=args.network)
    plt.title("Comparison of SincKAN and MLP Interpolations f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    fit_path = f"{args.datatype}_{args.network}_{args.seed}.png"
    plt.savefig(fit_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Residual related
    u_x = vmap(grad(net, argnums=1), (None, 0, None))(model, jnp.array(x_train)[:, 0], frozen_para)
    u_xx = vmap(grad(grad(net, argnums=1), argnums=1), (None, 0, None))(model, jnp.array(x_train)[:, 0], frozen_para)
    f = (u_xx / 100.0 + u_x)
    print(f"{(f**2).mean()}")
    np.savez("diff.npz", u_xx=np.array(u_xx), u_x=np.array(u_x), f=np.array(f))
    try:
        wandb.save("diff.npz")
    except Exception:
        pass

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(x_train, np.array(u_x), "r")
    ax[0].set_title("u_x")
    ax[1].plot(x_train, np.array(u_xx), "b-")
    ax[1].set_title("u_xx")
    ax[2].plot(x_train, np.array(f), "b-")
    ax[2].set_title("residual")
    diff_path = f"{args.datatype}_{args.network}_{args.seed}_diff.png"
    plt.savefig(diff_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # log images
    try:
        if os.path.exists(fit_path):
            wandb.log({"plots/fit": wandb.Image(fit_path)})
            wandb.save(fit_path)
        if os.path.exists(diff_path):
            wandb.log({"plots/diff": wandb.Image(diff_path)})
            wandb.save(diff_path)
    except Exception as e:
        print("wandb image logging failed:", e)

    # Timing reference
    T_ref = []
    for _ in range(10):
        t1 = time.time()
        _ = vmap(net, (None, 0, None))(model, jnp.array(x_train)[:, 0], frozen_para)
        t2 = time.time()
        T_ref.append(t2 - t1)

    avg_ref_time = float(np.mean(np.array(T_ref)))
    std_ref_time = float(np.std(np.array(T_ref)))
    print(f"ref_time: {avg_ref_time}")
    print(f"ref_time: {1 / avg_ref_time:.2e} ite/s")
    print(f"std of ref time: {std_ref_time}")

    wandb.log(
        {
            "perf/ref_time_sec": avg_ref_time,
            "perf/ref_steps_per_sec": float(1.0 / avg_ref_time) if avg_ref_time > 0 else 0.0,
            "perf/ref_time_std_sec": std_ref_time,
        }
    )


if __name__ == "__main__":
    seed = int(args.seed)
    np.random.seed(seed)
    key = random.PRNGKey(seed)

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    run_name = args.wandb_name or f"{args.datatype}-{args.network}-seed{args.seed}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        tags=tags,
    )

    train(key)
    eval(key)

    wandb.finish()
