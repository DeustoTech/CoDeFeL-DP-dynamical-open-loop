from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import struct
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiment_utils import (
    active_kkt_residual,
    floor_aware_entropy_step,
    floor_diagnostics,
    inequality_budget_kkt_residual,
    mirror_residual,
    privacy_epsilon,
    project_lower_bounded_simplex,
    projected_gradient_residual,
    variational_gap,
)


URLS = {
    "train_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

EXPECTED_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


@dataclass
class MLPShape:
    d: int
    h: int
    c: int

    @property
    def p(self) -> int:
        return self.d * self.h + self.h + self.h * self.c + self.c


@dataclass
class Problem:
    x_design: np.ndarray
    y_design: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    batches: list[np.ndarray]
    active_masks: np.ndarray
    theta_design_ref: np.ndarray
    theta_ref: np.ndarray
    eta: np.ndarray
    shape: MLPShape
    sensitivity: float
    l2: float
    clip: float
    rho_max: float
    q_min: float
    scenario: str
    phase_split: int | None
    pretrain_steps: int
    n_pretrain: int
    n_public: int
    n_design: int

    @property
    def T(self) -> int:
        return len(self.batches)


def download_file(url: str, path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        expected = EXPECTED_SHA256[path.name]
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(
                f"SHA-256 mismatch for downloaded {path}: expected {expected}, got {digest}"
            )
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_bytes(payload)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = EXPECTED_SHA256[path.name]
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {digest}")


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_fashion_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = {name: data_dir / Path(url).name for name, url in URLS.items()}
    for name, url in URLS.items():
        download_file(url, files[name])
    x_train = read_idx_images(files["train_images"])
    y_train = read_idx_labels(files["train_labels"])
    x_test = read_idx_images(files["test_images"])
    y_test = read_idx_labels(files["test_labels"])
    return x_train, y_train, x_test, y_test


def preprocess_images(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) / 255.0
    # 2x2 average pooling: 28x28 -> 14x14. This keeps the task image-based while
    # keeping the unrolled optimal-control experiment computationally manageable.
    x = x.reshape(x.shape[0], 14, 2, 14, 2).mean(axis=(2, 4))
    return x.reshape(x.shape[0], -1)


@lru_cache(maxsize=2)
def load_preprocessed_fashion_mnist(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_raw, y_train, x_test_raw, y_test = load_fashion_mnist(Path(data_dir))
    x_train = preprocess_images(x_train_raw)
    x_test = preprocess_images(x_test_raw)
    for array in [x_train, y_train, x_test, y_test]:
        array.setflags(write=False)
    return x_train, y_train, x_test, y_test


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    nonnegative = x >= 0.0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def make_problem(
    data_dir: Path,
    seed: int = 123,
    n_train: int = 8000,
    n_test: int = 3000,
    horizon: int = 45,
    batch_size: int = 192,
    hidden: int = 32,
    scenario: str = "standard",
    rho_max: float = 0.5,
    q_min: float = 1.0e-5,
    clip: float = 1.75,
) -> Problem:
    rng = np.random.default_rng(seed)
    xtr, ytr, xte, yte = load_preprocessed_fashion_mnist(str(data_dir.resolve()))

    # Public design and optional pretraining records are disjoint from the
    # private training split.
    train_perm = rng.permutation(len(xtr))
    train_idx = train_perm[:n_train]
    n_public = min(12000, len(xtr) - n_train)
    if n_public < n_train:
        raise ValueError("the public auxiliary split must contain at least n_train examples")
    public_idx = train_perm[n_train : n_train + n_public]
    n_pretrain = n_public if scenario == "lr_restart" else 0
    test_idx = rng.permutation(len(xte))[:n_test]
    design_idx = public_idx[:n_train]
    x_design, y_design = xtr[design_idx], ytr[design_idx]
    x_train, y_train = xtr[train_idx], ytr[train_idx]
    x_test, y_test = xte[test_idx], yte[test_idx]

    # Preprocessing statistics are estimated from the disjoint public/source
    # split rather than from the private optimization split.
    mean = xtr[public_idx].mean(axis=0)
    std = xtr[public_idx].std(axis=0) + 1e-6
    x_design = (x_design - mean) / std
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_pretrain = (xtr[public_idx] - mean) / std if n_pretrain else np.empty((0, x_train.shape[1]))
    y_pretrain = ytr[public_idx] if n_pretrain else np.empty(0, dtype=np.int64)

    T = horizon
    batches = [rng.choice(n_train, size=batch_size, replace=False) for _ in range(T)]
    shape = MLPShape(d=x_train.shape[1], h=hidden, c=10)
    active_masks, eta, phase_split = make_training_schedule(shape, T, scenario)
    l2 = 1.0e-4
    # Under replace-one adjacency, one contribution is replaced.  The
    # difference of the old and new clipped vectors has norm at most 2*clip,
    # so the averaged query has sensitivity 2*clip/batch_size.
    sensitivity = 2.0 * clip / batch_size
    if T * q_min >= rho_max:
        raise ValueError("q_min must satisfy T * q_min < rho_max")

    theta0 = init_theta(shape, rng)
    pretrain_steps = 300 if scenario == "lr_restart" else 0
    if pretrain_steps:
        theta0 = pretrain_theta(
            theta0,
            x_pretrain,
            y_pretrain,
            shape,
            l2,
            clip,
            steps=pretrain_steps,
            batch_size=batch_size,
            eta=0.05,
            seed=seed + 5000,
        )
    theta_design_ref = compute_reference(theta0, x_design, y_design, batches, active_masks, eta, shape, l2, clip)
    theta_ref = compute_reference(theta0, x_train, y_train, batches, active_masks, eta, shape, l2, clip)
    return Problem(
        x_design,
        y_design,
        x_train,
        y_train,
        x_test,
        y_test,
        batches,
        active_masks,
        theta_design_ref,
        theta_ref,
        eta,
        shape,
        sensitivity,
        l2,
        clip,
        rho_max,
        q_min,
        scenario,
        phase_split,
        pretrain_steps,
        n_pretrain,
        n_public,
        n_train,
    )


def make_training_schedule(shape: MLPShape, T: int, scenario: str) -> tuple[np.ndarray, np.ndarray, int | None]:
    if scenario == "standard":
        return np.ones((T, shape.p)), np.full(T, 0.15), None
    if scenario == "lr_restart":
        split = int(round(0.55 * T))
        eta = np.empty(T)
        eta[:split] = 0.02
        eta[split:] = 0.075
        return np.ones((T, shape.p)), eta, split
    raise ValueError(f"unknown scenario: {scenario}")


def init_theta(shape: MLPShape, rng: np.random.Generator) -> np.ndarray:
    W1 = rng.normal(scale=math.sqrt(1.0 / shape.d), size=(shape.d, shape.h))
    b1 = np.zeros(shape.h)
    W2 = rng.normal(scale=math.sqrt(1.0 / shape.h), size=(shape.h, shape.c))
    b2 = np.zeros(shape.c)
    return pack(W1[None], b1[None], W2[None], b2[None])[0]


def pretrain_theta(
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shape: MLPShape,
    l2: float,
    clip: float,
    steps: int,
    batch_size: int,
    eta: float,
    seed: int,
) -> np.ndarray:
    """Pretrain on a disjoint public auxiliary subset."""
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    for _ in range(steps):
        idx = rng.choice(len(x), size=batch_size, replace=False)
        grad = mlp_grad(theta[None, :], x[idx], y[idx], shape, l2, clip)[0]
        theta -= eta * grad
    return theta


def unpack(theta: np.ndarray, shape: MLPShape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.atleast_2d(theta)
    m = theta.shape[0]
    a = shape.d * shape.h
    b = a + shape.h
    c = b + shape.h * shape.c
    W1 = theta[:, :a].reshape(m, shape.d, shape.h)
    b1 = theta[:, a:b]
    W2 = theta[:, b:c].reshape(m, shape.h, shape.c)
    b2 = theta[:, c:]
    return W1, b1, W2, b2


def pack(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [W1.reshape(W1.shape[0], -1), b1, W2.reshape(W2.shape[0], -1), b2],
        axis=1,
    )


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=2, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=2, keepdims=True)


def unpack_mask(active_mask: np.ndarray | None, shape: MLPShape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(shape.p) if active_mask is None else np.asarray(active_mask, dtype=float)
    W1, b1, W2, b2 = unpack(mask[None, :], shape)
    return W1[0], b1[0], W2[0], b2[0], mask


def mlp_data_terms(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shape: MLPShape,
) -> tuple[np.ndarray, ...]:
    theta = np.atleast_2d(theta)
    W1, b1, W2, b2 = unpack(theta, shape)
    z1 = np.einsum("bd,mdh->mbh", x, W1) + b1[:, None, :]
    activation_prime = sigmoid(z1)
    activation_second = activation_prime * (1.0 - activation_prime)
    a1 = softplus(z1)
    logits = np.einsum("mbh,mhc->mbc", a1, W2) + b2[:, None, :]
    probs = softmax_logits(logits)
    delta = probs.copy()
    m, bsz = theta.shape[0], x.shape[0]
    delta[np.arange(m)[:, None], np.arange(bsz)[None, :], y[None, :]] -= 1.0
    dz1 = np.einsum("mbc,mhc->mbh", delta, W2) * activation_prime
    return W1, b1, W2, b2, activation_prime, activation_second, a1, probs, delta, dz1


def per_example_clip_scales(
    x: np.ndarray,
    a1: np.ndarray,
    delta: np.ndarray,
    dz1: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    mW1, mb1, mW2, mb2 = masks
    all_active = all(np.all(mask == 1.0) for mask in masks)
    if all_active:
        norm_sq = (
            np.sum(x * x, axis=1)[None, :] + 1.0
        ) * np.sum(dz1 * dz1, axis=2)
        norm_sq += (
            np.sum(a1 * a1, axis=2) + 1.0
        ) * np.sum(delta * delta, axis=2)
    else:
        norm_sq = np.einsum("bd,mbh,dh->mb", x * x, dz1 * dz1, mW1)
        norm_sq += np.einsum("mbh,h->mb", dz1 * dz1, mb1)
        norm_sq += np.einsum("mbh,mbc,hc->mb", a1 * a1, delta * delta, mW2)
        norm_sq += np.einsum("mbc,c->mb", delta * delta, mb2)
    scales = 1.0 / np.sqrt(1.0 + norm_sq / (clip * clip))
    return scales, norm_sq


def mlp_grad(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shape: MLPShape,
    l2: float,
    clip: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Average smoothly clipped per-example MLP gradients."""
    theta = np.atleast_2d(theta)
    _, _, _, _, _, _, a1, _, delta, dz1 = mlp_data_terms(theta, x, y, shape)
    mW1, mb1, mW2, mb2, mask = unpack_mask(active_mask, shape)
    scales, _ = per_example_clip_scales(x, a1, delta, dz1, (mW1, mb1, mW2, mb2), clip)
    bsz = x.shape[0]
    gW1 = np.einsum("mb,bd,mbh->mdh", scales, x, dz1) / bsz
    gb1 = np.einsum("mb,mbh->mh", scales, dz1) / bsz
    gW2 = np.einsum("mb,mbh,mbc->mhc", scales, a1, delta) / bsz
    gb2 = np.einsum("mb,mbc->mc", scales, delta) / bsz
    data_grad = pack(
        gW1 * mW1[None, :, :],
        gb1 * mb1[None, :],
        gW2 * mW2[None, :, :],
        gb2 * mb2[None, :],
    )
    return data_grad + l2 * theta * mask[None, :]


def mlp_grad_vjp(
    theta: np.ndarray,
    cotangent: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shape: MLPShape,
    l2: float,
    clip: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Exact VJP of the Softplus, smoothly clipped per-example gradient."""
    theta = np.atleast_2d(theta)
    cotangent = np.atleast_2d(cotangent)
    W1, _, W2, _, activation_prime, activation_second, a1, probs, delta, dz1 = mlp_data_terms(theta, x, y, shape)
    mW1, mb1, mW2, mb2, mask = unpack_mask(active_mask, shape)
    scales, norm_sq = per_example_clip_scales(x, a1, delta, dz1, (mW1, mb1, mW2, mb2), clip)
    all_active = bool(np.all(mask == 1.0))

    pW1, pb1, pW2, pb2 = unpack(cotangent, shape)
    pW1a = pW1 * mW1[None, :, :]
    pb1a = pb1 * mb1[None, :]
    pW2a = pW2 * mW2[None, :, :]
    pb2a = pb2 * mb2[None, :]
    inner = np.einsum("bd,mbh,mdh->mb", x, dz1, pW1a)
    inner += np.einsum("mbh,mh->mb", dz1, pb1a)
    inner += np.einsum("mbh,mbc,mhc->mb", a1, delta, pW2a)
    inner += np.einsum("mbc,mc->mb", delta, pb2a)

    correction = scales * inner / (clip * clip + norm_sq)

    # Only the forward directional products are needed below.  Contract the
    # clipped cotangent analytically instead of materializing M x B x d x h
    # and M x B x h x c tensors.
    z1_dot = scales[:, :, None] * (
        np.einsum("bd,mdh->mbh", x, pW1a) + pb1a[:, None, :]
    )
    if all_active:
        masked_x_norm = np.sum(x * x, axis=1)[:, None] + 1.0
    else:
        masked_x_norm = np.einsum("bd,dh->bh", x * x, mW1) + mb1[None, :]
    z1_dot -= correction[:, :, None] * dz1 * masked_x_norm[None, :, :]
    a1_dot = activation_prime * z1_dot
    logits_dot = np.einsum("mbh,mhc->mbc", a1_dot, W2)
    logits_dot += scales[:, :, None] * (
        np.einsum("mbh,mhc->mbc", a1, pW2a) + pb2a[:, None, :]
    )
    if all_active:
        masked_activation_norm = np.sum(a1 * a1, axis=2, keepdims=True) + 1.0
    else:
        masked_activation_norm = np.einsum("mbh,hc->mbc", a1 * a1, mW2)
        masked_activation_norm += mb2[None, None, :]
    logits_dot -= correction[:, :, None] * delta * masked_activation_norm
    delta_dot = probs * (
        logits_dot - np.sum(probs * logits_dot, axis=2, keepdims=True)
    )
    backprop_signal = np.einsum("mbc,mhc->mbh", delta, W2)
    backprop_dot = np.einsum("mbc,mhc->mbh", delta_dot, W2)
    backprop_dot += scales[:, :, None] * np.einsum("mbc,mhc->mbh", delta, pW2a)
    if all_active:
        masked_delta_norm = np.sum(delta * delta, axis=2, keepdims=True)
    else:
        masked_delta_norm = np.einsum("mbc,hc->mbh", delta * delta, mW2)
    backprop_dot -= correction[:, :, None] * a1 * masked_delta_norm
    dz1_dot = backprop_dot * activation_prime
    dz1_dot += backprop_signal * activation_second * z1_dot

    bsz = x.shape[0]
    hW1 = np.einsum("bd,mbh->mdh", x, dz1_dot) / bsz
    hb1 = np.mean(dz1_dot, axis=1)
    hW2 = (
        np.einsum("mbh,mbc->mhc", a1_dot, delta)
        + np.einsum("mbh,mbc->mhc", a1, delta_dot)
    ) / bsz
    hb2 = np.mean(delta_dot, axis=1)
    return pack(hW1, hb1, hW2, hb2) + l2 * cotangent * mask[None, :]


def mlp_loss_acc(theta: np.ndarray, x: np.ndarray, y: np.ndarray, shape: MLPShape, l2: float, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    theta = np.atleast_2d(theta)
    m = theta.shape[0]
    losses = np.zeros(m)
    correct = np.zeros(m)
    total = 0
    W1, b1, W2, b2 = unpack(theta, shape)
    reg = 0.5 * l2 * np.sum(theta * theta, axis=1)
    for start in range(0, x.shape[0], batch):
        xb = x[start : start + batch]
        yb = y[start : start + batch]
        z1 = np.einsum("bd,mdh->mbh", xb, W1) + b1[:, None, :]
        a1 = softplus(z1)
        logits = np.einsum("mbh,mhc->mbc", a1, W2) + b2[:, None, :]
        z = logits - logits.max(axis=2, keepdims=True)
        logsum = np.log(np.exp(z).sum(axis=2)) + logits.max(axis=2)
        losses += np.sum(logsum - logits[:, np.arange(len(yb)), yb], axis=1)
        pred = logits.argmax(axis=2)
        correct += np.sum(pred == yb[None, :], axis=1)
        total += len(yb)
    return losses / total + reg, correct / total


def compute_reference(theta0: np.ndarray, x: np.ndarray, y: np.ndarray, batches: list[np.ndarray], active_masks: np.ndarray, eta: np.ndarray, shape: MLPShape, l2: float, clip: float) -> np.ndarray:
    theta = np.zeros((len(batches) + 1, shape.p))
    theta[0] = theta0
    for t, idx in enumerate(batches):
        grad = mlp_grad(theta[t][None], x[idx], y[idx], shape, l2, clip, active_masks[t])[0]
        theta[t + 1] = theta[t] - eta[t] * grad
    return theta


def objective_and_grad(problem: Problem, q: np.ndarray, xis: np.ndarray, theta0: np.ndarray) -> tuple[float, np.ndarray]:
    T, M, P = xis.shape
    theta = np.zeros((T + 1, M, P))
    theta[0] = theta0[None, :]
    for t, idx in enumerate(problem.batches):
        grad = mlp_grad(
            theta[t],
            problem.x_design[idx],
            problem.y_design[idx],
            problem.shape,
            problem.l2,
            problem.clip,
            problem.active_masks[t],
        )
        amp = problem.sensitivity * problem.eta[t] / math.sqrt(2.0 * q[t])
        active_noise = problem.active_masks[t][None, :] * xis[t]
        theta[t + 1] = theta[t] - problem.eta[t] * grad - amp * active_noise
    diff = theta[T] - problem.theta_design_ref[T][None, :]
    obj = float(np.mean(np.sum(diff * diff, axis=1)))
    p = 2.0 * diff
    h = np.zeros(T)
    for t in range(T - 1, -1, -1):
        active_p = problem.active_masks[t][None, :] * p
        h[t] = np.mean(problem.sensitivity * problem.eta[t] * (2.0 * q[t]) ** (-1.5) * np.sum(active_p * xis[t], axis=1))
        idx = problem.batches[t]
        vjp = mlp_grad_vjp(
            theta[t],
            p,
            problem.x_design[idx],
            problem.y_design[idx],
            problem.shape,
            problem.l2,
            problem.clip,
            problem.active_masks[t],
        )
        p = p - problem.eta[t] * vjp
    return obj, h


def objective_only(problem: Problem, q: np.ndarray, xis: np.ndarray, theta0: np.ndarray) -> float:
    _, M, _ = xis.shape
    theta = np.repeat(theta0[None, :], M, axis=0)
    for t, idx in enumerate(problem.batches):
        grad = mlp_grad(
            theta,
            problem.x_design[idx],
            problem.y_design[idx],
            problem.shape,
            problem.l2,
            problem.clip,
            problem.active_masks[t],
        )
        amp = problem.sensitivity * problem.eta[t] / math.sqrt(2.0 * q[t])
        active_noise = problem.active_masks[t][None, :] * xis[t]
        theta = theta - problem.eta[t] * grad - amp * active_noise
    diff = theta - problem.theta_design_ref[problem.T][None, :]
    return float(np.mean(np.sum(diff * diff, axis=1)))


def simulate_eval_many(
    problem: Problem,
    schedules: list[np.ndarray],
    theta0: np.ndarray,
    seed: int,
    paths: int = 96,
    chunk: int = 16,
) -> list[dict[str, float]]:
    """Evaluate schedules in one batch with common Gaussian innovations."""
    q_matrix = np.asarray(schedules, dtype=float)
    if q_matrix.ndim != 2 or q_matrix.shape[1] != problem.T:
        raise ValueError("every schedule must have one coordinate per time step")
    n_methods = q_matrix.shape[0]
    rng = np.random.default_rng(seed)
    terminal = np.empty((n_methods, paths))
    test_acc = np.empty((n_methods, paths))
    train_acc = np.empty((n_methods, paths))
    test_loss = np.empty((n_methods, paths))
    for start in range(0, paths, chunk):
        m = min(chunk, paths - start)
        theta = np.broadcast_to(theta0[None, None, :], (n_methods, m, problem.shape.p)).copy()
        for t, idx in enumerate(problem.batches):
            flat_theta = theta.reshape(n_methods * m, problem.shape.p)
            grad = mlp_grad(
                flat_theta,
                problem.x_train[idx],
                problem.y_train[idx],
                problem.shape,
                problem.l2,
                problem.clip,
                problem.active_masks[t],
            ).reshape(n_methods, m, problem.shape.p)
            amp = problem.sensitivity * problem.eta[t] / np.sqrt(2.0 * q_matrix[:, t])
            innovation = problem.active_masks[t][None, :] * rng.normal(size=(m, problem.shape.p))
            theta = theta - problem.eta[t] * grad - amp[:, None, None] * innovation[None, :, :]
        diff = theta - problem.theta_ref[problem.T][None, None, :]
        terminal[:, start : start + m] = np.sum(diff * diff, axis=2)
        flat_theta = theta.reshape(n_methods * m, problem.shape.p)
        tl, ta = mlp_loss_acc(flat_theta, problem.x_test, problem.y_test, problem.shape, problem.l2)
        _, tra = mlp_loss_acc(
            flat_theta,
            problem.x_train[:2000],
            problem.y_train[:2000],
            problem.shape,
            problem.l2,
        )
        test_loss[:, start : start + m] = tl.reshape(n_methods, m)
        test_acc[:, start : start + m] = ta.reshape(n_methods, m)
        train_acc[:, start : start + m] = tra.reshape(n_methods, m)
    return [
        {
            "terminal_mean": float(np.mean(terminal[index])),
            "terminal_std": float(np.std(terminal[index])),
            "test_loss": float(np.mean(test_loss[index])),
            "test_acc": float(np.mean(test_acc[index])),
            "test_acc_std": float(np.std(test_acc[index])),
            "train_acc_2k": float(np.mean(train_acc[index])),
        }
        for index in range(n_methods)
    ]


def simulate_eval(
    problem: Problem,
    q: np.ndarray,
    theta0: np.ndarray,
    seed: int,
    paths: int = 96,
    chunk: int = 16,
) -> dict[str, float]:
    return simulate_eval_many(problem, [q], theta0, seed, paths, chunk)[0]


def run_pb(
    problem: Problem,
    theta0: np.ndarray,
    xis: np.ndarray,
    K: int,
    alpha: float,
    record_trace: bool = True,
    kkt_tolerance: float | None = None,
    max_iterations: int | None = None,
) -> dict:
    if kkt_tolerance is not None:
        if kkt_tolerance <= 0.0:
            raise ValueError("kkt_tolerance must be positive")
        if max_iterations is None:
            max_iterations = 10 * K
        if K < 1 or max_iterations < K:
            raise ValueError("max_iterations must be at least K >= 1")
        return _run_pb_to_kkt(
            problem,
            theta0,
            xis,
            K,
            max_iterations,
            alpha,
            kkt_tolerance,
            record_trace,
        )

    print(f"running PB-AMD alpha={alpha:g}", flush=True)
    T = problem.T
    q = np.full(T, problem.rho_max / T)
    start = time.perf_counter()
    hist = []
    trace = []
    best_obj = float("inf")
    best_q = None
    max_feasibility_error = 0.0
    for k in range(K):
        obj, h = objective_and_grad(problem, q, xis, theta0)
        if obj < best_obj:
            best_obj = obj
            best_q = q.copy()
        if record_trace and (k % 5 == 0 or k == K - 1):
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": obj,
                "rho_spent": float(q.sum()),
                "rho_violation": max(float(q.sum() - problem.rho_max), 0.0),
                "rho_slack": max(float(problem.rho_max - q.sum()), 0.0),
                "optimizer_residual": mirror_residual(
                    q,
                    h,
                    alpha,
                    problem.rho_max,
                    problem.q_min,
                ),
                "kkt_residual": active_kkt_residual(q, h, problem.q_min),
            })
        q, _, feasibility_error = floor_aware_entropy_step(
            q,
            h,
            alpha,
            problem.rho_max,
            problem.q_min,
        )
        max_feasibility_error = max(max_feasibility_error, feasibility_error)
        if record_trace and (k % 5 == 0 or k == K - 1):
            hist.append((k, obj, float(q.sum()), feasibility_error))
    obj = objective_only(problem, q, xis, theta0)
    if obj < best_obj:
        best_obj = obj
        best_q = q.copy()
    elapsed = time.perf_counter() - start
    assert best_q is not None
    print(f"finished PB-AMD alpha={alpha:g} in {elapsed:.1f}s", flush=True)
    return {
        "method": f"PB-AMD alpha={alpha:g}",
        "family": "PB-AMD",
        "alpha": alpha,
        "q": best_q,
        "design_objective": best_obj,
        "time_sec": elapsed,
        "history": hist,
        "trace": trace,
        "max_prox_feasibility_error": max_feasibility_error,
        "iterations": K,
    }


def _run_pb_to_kkt(
    problem: Problem,
    theta0: np.ndarray,
    xis: np.ndarray,
    minimum_iterations: int,
    max_iterations: int,
    alpha: float,
    kkt_tolerance: float,
    record_trace: bool,
) -> dict:
    """Run at least ``minimum_iterations`` updates and certify the returned iterate."""
    print(
        f"running PB-AMD alpha={alpha:g} to KKT tolerance {kkt_tolerance:g}",
        flush=True,
    )
    q = np.full(problem.T, problem.rho_max / problem.T)
    history = []
    trace = []
    best_obj = float("inf")
    best_q = None
    best_kkt = float("inf")
    max_feasibility_error = 0.0
    converged = False
    iterations = 0
    start = time.perf_counter()

    for k in range(max_iterations + 1):
        obj, gradient = objective_and_grad(problem, q, xis, theta0)
        candidate_kkt = active_kkt_residual(q, gradient, problem.q_min)
        if obj < best_obj:
            best_obj = obj
            best_q = q.copy()
            best_kkt = candidate_kkt

        should_record = (
            record_trace
            and (
                k % 5 == 0
                or k == minimum_iterations
                or k == max_iterations
                or (k >= minimum_iterations and best_kkt <= kkt_tolerance)
            )
        )
        if should_record:
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": obj,
                "rho_spent": float(q.sum()),
                "rho_violation": max(float(q.sum() - problem.rho_max), 0.0),
                "rho_slack": max(float(problem.rho_max - q.sum()), 0.0),
                "optimizer_residual": mirror_residual(
                    q,
                    gradient,
                    alpha,
                    problem.rho_max,
                    problem.q_min,
                ),
                "kkt_residual": candidate_kkt,
            })

        if k >= minimum_iterations and best_kkt <= kkt_tolerance:
            converged = True
            iterations = k
            break
        if k == max_iterations:
            iterations = k
            break

        q, _, feasibility_error = floor_aware_entropy_step(
            q,
            gradient,
            alpha,
            problem.rho_max,
            problem.q_min,
        )
        max_feasibility_error = max(max_feasibility_error, feasibility_error)
        if should_record:
            history.append((k, obj, float(q.sum()), feasibility_error))

    elapsed = time.perf_counter() - start
    assert best_q is not None
    print(
        f"finished PB-AMD alpha={alpha:g} after {iterations} updates "
        f"with KKT residual {best_kkt:.3e} in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "method": f"PB-AMD alpha={alpha:g}",
        "family": "PB-AMD",
        "alpha": alpha,
        "q": best_q,
        "design_objective": best_obj,
        "time_sec": elapsed,
        "history": history,
        "trace": trace,
        "max_prox_feasibility_error": max_feasibility_error,
        "iterations": iterations,
        "gradient_evaluations": iterations + 1,
        "kkt_tolerance": kkt_tolerance,
        "certified_kkt_residual": best_kkt,
        "converged": converged,
        "stopping_reason": "kkt_tolerance" if converged else "maximum_iterations",
    }


def run_projected_gradient(
    problem: Problem,
    theta0: np.ndarray,
    xis: np.ndarray,
    K: int,
    alpha: float,
    record_trace: bool = True,
) -> dict:
    """Euclidean projected gradient on the exact lower-bounded budget set."""
    print(f"running PGD alpha={alpha:g}", flush=True)
    q = np.full(problem.T, problem.rho_max / problem.T)
    history = []
    trace = []
    best_obj = float("inf")
    best_q = None
    start = time.perf_counter()
    for k in range(K):
        obj, gradient = objective_and_grad(problem, q, xis, theta0)
        if obj < best_obj:
            best_obj = obj
            best_q = q.copy()
        if record_trace and (k % 5 == 0 or k == K - 1):
            residual = projected_gradient_residual(
                q,
                gradient,
                alpha,
                problem.rho_max,
                problem.q_min,
            )
            history.append((
                k,
                obj,
                float(q.sum()),
                residual,
            ))
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": obj,
                "rho_spent": float(q.sum()),
                "rho_violation": max(float(q.sum() - problem.rho_max), 0.0),
                "rho_slack": max(float(problem.rho_max - q.sum()), 0.0),
                "optimizer_residual": residual,
                "kkt_residual": active_kkt_residual(q, gradient, problem.q_min),
            })
        q = project_lower_bounded_simplex(
            q - alpha * gradient,
            problem.rho_max,
            problem.q_min,
        )
    obj = objective_only(problem, q, xis, theta0)
    if obj < best_obj:
        best_obj = obj
        best_q = q.copy()
    elapsed = time.perf_counter() - start
    assert best_q is not None
    print(f"finished PGD alpha={alpha:g} in {elapsed:.1f}s", flush=True)
    return {
        "method": f"PGD alpha={alpha:g}",
        "family": "PGD",
        "alpha": alpha,
        "beta": "",
        "lr": "",
        "q": best_q,
        "design_objective": best_obj,
        "history": history,
        "trace": trace,
        "time_sec": elapsed,
        "max_prox_feasibility_error": abs(float(best_q.sum()) - problem.rho_max),
        "iterations": K,
    }


def run_penalty(
    problem: Problem,
    theta0: np.ndarray,
    xis: np.ndarray,
    K: int,
    beta: float,
    lr: float,
    initial_q: np.ndarray | None = None,
    initialization: str = "uniform",
    compute_diagnostics: bool = True,
    record_trace: bool = True,
) -> dict:
    print(f"running penalty beta={beta:g}", flush=True)
    T = problem.T
    q0 = (
        np.full(T, problem.rho_max / T)
        if initial_q is None
        else np.asarray(initial_q, dtype=float).copy()
    )
    if q0.shape != (T,) or np.any(q0 <= 0.0):
        raise ValueError("the penalty initialization must be a positive schedule")
    v = -np.log(2.0 * q0)
    m = np.zeros(T)
    s = np.zeros(T)
    b1, b2 = 0.9, 0.999
    start = time.perf_counter()
    hist = []
    trace = []
    best_penalized = float("inf")
    best_q = None
    best_obj = float("inf")
    for k in range(K):
        q = np.clip(0.5 * np.exp(-v), 1e-8, 2.0 * problem.rho_max)
        obj, h = objective_and_grad(problem, q, xis, theta0)
        phi = float(q.sum() - problem.rho_max)
        grad_v = -q * h - beta * max(phi, 0.0) * q
        m = b1 * m + (1.0 - b1) * grad_v
        s = b2 * s + (1.0 - b2) * grad_v * grad_v
        v = v - lr * (m / (1.0 - b1 ** (k + 1))) / (np.sqrt(s / (1.0 - b2 ** (k + 1))) + 1e-8)
        v = np.clip(v, -math.log(2.0 * 2.0 * problem.rho_max), -math.log(2.0 * 1e-8))
        penalized = obj + 0.5 * beta * max(phi, 0.0) ** 2
        if penalized < best_penalized:
            best_penalized = penalized
            best_q = q.copy()
            best_obj = obj
        if record_trace and (k % 5 == 0 or k == K - 1):
            hist.append((k, penalized, float(q.sum()), max(phi, 0.0)))
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": penalized,
                "rho_spent": float(q.sum()),
                "rho_violation": max(phi, 0.0),
                "rho_slack": max(-phi, 0.0),
                "optimizer_residual": float(np.linalg.norm(grad_v)),
                "kkt_residual": inequality_budget_kkt_residual(
                    q,
                    h,
                    problem.rho_max,
                ),
            })
    q = np.clip(0.5 * np.exp(-v), 1e-8, 2.0 * problem.rho_max)
    obj = objective_only(problem, q, xis, theta0)
    phi = float(q.sum() - problem.rho_max)
    penalized = obj + 0.5 * beta * max(phi, 0.0) ** 2
    if penalized < best_penalized:
        best_q = q.copy()
        best_obj = obj
    elapsed = time.perf_counter() - start
    assert best_q is not None
    q = best_q
    phi = float(q.sum() - problem.rho_max)
    optimizer_grad_norm: float | str = ""
    if compute_diagnostics:
        _, h = objective_and_grad(problem, q, xis, theta0)
        penalized_gradient = -q * h - beta * max(phi, 0.0) * q
        optimizer_grad_norm = float(np.linalg.norm(penalized_gradient))
    print(f"finished penalty beta={beta:g} in {elapsed:.1f}s", flush=True)
    return {
        "method": f"Penalty beta={beta:g}",
        "family": "Penalty",
        "q": q,
        "design_objective": best_obj,
        "time_sec": elapsed,
        "history": hist,
        "trace": trace,
        "beta": beta,
        "lr": lr,
        "optimizer_grad_norm": optimizer_grad_norm,
        "iterations": K,
        "initialization": initialization,
    }


def summarize_schedule(q: np.ndarray, rho: float) -> dict[str, float]:
    T = len(q)
    early = q[: T // 3]
    middle = q[T // 3 : 2 * T // 3]
    late = q[2 * T // 3 :]
    return {
        "rho_spent": float(q.sum()),
        "rho_violation": max(float(q.sum() - rho), 0.0),
        "rho_slack": max(float(rho - q.sum()), 0.0),
        "min_q": float(q.min()),
        "max_q": float(q.max()),
        "early_budget": float(early.sum()),
        "middle_budget": float(middle.sum()),
        "late_budget": float(late.sum()),
        "late_over_early": float(late.mean() / max(float(early.mean()), 1e-12)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/fashion_mnist_mlp_schedule")
    parser.add_argument("--data-dir", default="data/fashion_mnist")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--M-opt", type=int, default=8)
    parser.add_argument("--M-val", type=int, default=48)
    parser.add_argument("--M-eval", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--scenario", choices=["standard", "lr_restart"], default="standard")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--q-min", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.75)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--alphas", default="0.0005,0.001,0.002,0.005,0.01")
    parser.add_argument("--pgd-alphas", default="3e-6,1e-5,3e-5,1e-4")
    parser.add_argument(
        "--penalty-grid",
        default="1:0.035,3:0.032,10:0.025,100:0.017,1000:0.01,10000:0.006",
        help="comma-separated beta:learning-rate pairs; use an empty string to skip penalties",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = make_problem(
        Path(args.data_dir),
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        horizon=args.horizon,
        batch_size=args.batch_size,
        hidden=args.hidden,
        scenario=args.scenario,
        rho_max=args.rho,
        q_min=args.q_min,
        clip=args.clip,
    )
    theta0 = problem.theta_design_ref[0]
    opt_seed = args.seed + 77
    val_seed = args.seed + 177
    eval_seed = args.seed + 1000
    xis = np.random.default_rng(opt_seed).normal(size=(problem.T, args.M_opt, problem.shape.p))
    xis_val = np.random.default_rng(val_seed).normal(size=(problem.T, args.M_val, problem.shape.p))

    alphas = [float(value) for value in args.alphas.split(",") if value]
    pgd_alphas = [float(value) for value in args.pgd_alphas.split(",") if value]
    penalty_grid = []
    for item in args.penalty_grid.split(","):
        if item:
            beta, lr = item.split(":", maxsplit=1)
            penalty_grid.append((float(beta), float(lr)))
    methods = [run_pb(problem, theta0, xis, args.K, alpha=alpha) for alpha in alphas]
    methods.extend(
        run_projected_gradient(problem, theta0, xis, args.K, alpha=alpha)
        for alpha in pgd_alphas
    )
    methods.extend(
        run_penalty(problem, theta0, xis, args.K, beta=beta, lr=lr)
        for beta, lr in penalty_grid
    )
    q_uniform = np.full(problem.T, problem.rho_max / problem.T)
    methods.append({"method": "Uniform feasible", "family": "Uniform", "q": q_uniform, "design_objective": objective_and_grad(problem, q_uniform, xis, theta0)[0], "time_sec": 0.0, "history": []})

    for method in methods:
        method["validation_objective"] = objective_only(problem, method["q"], xis_val, theta0)

    best_pb = min(
        (method for method in methods if method["family"] == "PB-AMD"),
        key=lambda method: method["validation_objective"],
    )
    best_pgd = min(
        (method for method in methods if method["family"] == "PGD"),
        key=lambda method: method["validation_objective"],
    )
    penalties = [method for method in methods if method["family"] == "Penalty"]
    feasibility_tol = 1e-10 * max(1.0, problem.rho_max)
    feasible_penalties = [
        method for method in penalties
        if float(np.sum(method["q"])) <= problem.rho_max + feasibility_tol
    ]
    best_penalty = min(
        feasible_penalties or penalties,
        key=lambda method: method["validation_objective"],
        default=None,
    )

    evaluation_metrics = simulate_eval_many(
        problem,
        [m["q"] for m in methods],
        theta0,
        seed=eval_seed,
        paths=args.M_eval,
    )
    reference_test_loss, reference_test_acc = mlp_loss_acc(
        problem.theta_ref[-1][None, :],
        problem.x_test,
        problem.y_test,
        problem.shape,
        problem.l2,
    )
    rows = []
    for m, evals in zip(methods, evaluation_metrics):
        print(f"evaluating {m['method']}", flush=True)
        q = m["q"]
        exact_feasible = (
            abs(float(q.sum()) - problem.rho_max) <= 1e-10
            and float(q.min()) >= problem.q_min - 1e-12
        )
        report_stationarity = exact_feasible and m["family"] in {"PB-AMD", "PGD"}
        if report_stationarity:
            _, saa_gradient = objective_and_grad(problem, q, xis, theta0)
            _, validation_gradient = objective_and_grad(problem, q, xis_val, theta0)
        phase_ratio = ""
        if problem.phase_split is not None:
            phase_ratio = float(q[problem.phase_split :].mean() / max(float(q[: problem.phase_split].mean()), 1e-12))
        sched = summarize_schedule(q, problem.rho_max)
        rows.append({
            "method": m["method"],
            "family": m["family"],
            "alpha": m.get("alpha", ""),
            "beta": m.get("beta", ""),
            "lr": m.get("lr", ""),
            "initialization": m.get("initialization", ""),
            "design_objective": float(m["design_objective"]),
            "validation_objective": float(m["validation_objective"]),
            **evals,
            "reference_test_loss": float(reference_test_loss[0]),
            "reference_test_acc": float(reference_test_acc[0]),
            **sched,
            "delta": args.delta,
            "epsilon_target": privacy_epsilon(problem.rho_max, args.delta),
            "epsilon_actual": privacy_epsilon(sched["rho_spent"], args.delta),
            "phase_late_over_early": phase_ratio,
            "saa_variational_gap": variational_gap(q, saa_gradient, problem.rho_max, problem.q_min) if report_stationarity else "",
            "saa_mirror_residual": mirror_residual(q, saa_gradient, float(m["alpha"]), problem.rho_max, problem.q_min) if m["family"] == "PB-AMD" else "",
            "saa_projected_residual": projected_gradient_residual(q, saa_gradient, float(m["alpha"]), problem.rho_max, problem.q_min) if m["family"] == "PGD" else "",
            "saa_kkt_residual": active_kkt_residual(q, saa_gradient, problem.q_min) if report_stationarity else "",
            "validation_variational_gap": variational_gap(q, validation_gradient, problem.rho_max, problem.q_min) if report_stationarity else "",
            **floor_diagnostics(q, problem.q_min),
            "optimizer_grad_norm": m.get("optimizer_grad_norm", ""),
            "iterations": m.get("iterations", ""),
            "max_prox_feasibility_error": m.get("max_prox_feasibility_error", ""),
            "time_sec": float(m["time_sec"]),
        })
    write_csv(out_dir / "summary.csv", rows)
    trace_rows = []
    for method in methods:
        for trace in method.get("trace", []):
            trace_rows.append({"method": method["method"], "family": method["family"], **trace})
    if trace_rows:
        write_csv(out_dir / "optimization_traces.csv", trace_rows)
    with (out_dir / "schedules.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [m["method"] for m in methods])
        for t in range(problem.T):
            w.writerow([t] + [float(m["q"][t]) for m in methods])
    with (out_dir / "config.json").open("w") as f:
        json.dump({
            "dataset": "Fashion-MNIST",
            "scenario": problem.scenario,
            "preprocessing": "2x2 average pooling to 14x14",
            "model": f"Softplus MLP {problem.shape.d}-{problem.shape.h}-{problem.shape.c}",
            "parameters": problem.shape.p,
            "n_public_design": problem.n_design,
            "n_train": int(problem.x_train.shape[0]),
            "n_test": int(problem.x_test.shape[0]),
            "batch_size": args.batch_size,
            "T": problem.T,
            "rho_max": problem.rho_max,
            "q_min": problem.q_min,
            "K": args.K,
            "M_opt": args.M_opt,
            "M_val": args.M_val,
            "M_eval": args.M_eval,
            "alphas": alphas,
            "pgd_alphas": pgd_alphas,
            "penalty_grid": penalty_grid,
            "eta": float(problem.eta[0]),
            "eta_min": float(problem.eta.min()),
            "eta_max": float(problem.eta.max()),
            "sensitivity": problem.sensitivity,
            "clip": problem.clip,
            "clipping_operator": "g / sqrt(1 + ||g||^2 / clip^2)",
            "aggregation": "average of smoothly clipped per-example gradients",
            "sensitivity_model": "replacement adjacency; public calibration bound 2 * clip / batch_size",
            "privacy_charge_interpretation": "q_t is a certified upper bound; exact worst-case charge is (Delta_exact / Delta_hat)^2 q_t",
            "privacy_amplification": "none; accountant conditions on public batches and uses direct adaptive composition",
            "noise_gradient_std": "sensitivity / sqrt(2 * q_t)",
            "schedule_design_data": "disjoint public auxiliary split",
            "selection": "independent public validation Gaussian paths",
            "final_evaluation": "fresh private-training Gaussian paths shared across methods",
            "optimization_seed": opt_seed,
            "validation_seed": val_seed,
            "evaluation_seed": eval_seed,
            "selected_pb_method": best_pb["method"],
            "selected_pgd_method": best_pgd["method"],
            "selected_penalty_method": best_penalty["method"] if best_penalty else None,
            "delta": args.delta,
            "epsilon_target": privacy_epsilon(problem.rho_max, args.delta),
            "q_min_over_uniform": problem.q_min / (problem.rho_max / problem.T),
            "test_set_role": "public evaluation only; never used for schedule or hyperparameter selection",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "fashion_mnist_sha256": EXPECTED_SHA256,
            "phase_split": problem.phase_split,
            "pretrain_steps": problem.pretrain_steps,
            "n_pretrain_disjoint_from_private": problem.n_pretrain,
            "n_public_pretrain_records": problem.n_pretrain,
            "pretrain_data_relation": (
                "public auxiliary split; disjoint from private training and overlapping the public design subset"
                if problem.pretrain_steps
                else "not used in the standard protocol"
            ),
            "n_public_preprocessing": problem.n_public,
            "pretrain_eta": 0.05 if problem.pretrain_steps else None,
            "active_parameters_first_phase": int(problem.active_masks[0].sum()),
            "active_parameters_second_phase": int(problem.active_masks[-1].sum()),
            "gaussian_noise_dimension": problem.shape.p,
            "active_mask": "deterministic all-ones mask",
        }, f, indent=2)
    print(json.dumps({"out_dir": str(out_dir), "summary": str(out_dir / "summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
