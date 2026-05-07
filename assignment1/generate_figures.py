"""
Generate NTT performance figures for the assignment report.

Produces three figures:
  1. Throughput heatmap over (logn, batch)
  2. Throughput sweep over logn at fixed batch sizes
  3. Compile time vs median latency scatter

Usage:
    uv run python generate_figures.py
"""

from __future__ import annotations

import os
import time

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
import jax.numpy as jnp
import numpy as np

import provided
import student

# ── Configuration ────────────────────────────────────────────────────────────

LOGN_RANGE = list(range(8, 15))        # 256 … 16384
BATCH_RANGE = [1, 2, 4, 8, 16, 32]
RUNS = 20
WARMUP = 5
BIT_LENGTH = 31
SEED = 42

# ── Benchmark core ───────────────────────────────────────────────────────────

def bench(N, batch, q, psi, runs, warmup, rng):
    x_np = rng.integers(0, q, size=(batch, N), dtype=np.int64)
    x = jnp.asarray(x_np, dtype=jnp.uint32)

    psi_powers, twiddles = provided.precompute_tables(N, q, psi)
    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles = jnp.asarray(twiddles, dtype=jnp.uint32)

    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    fn = jax.jit(
        lambda z: student.ntt(z, q=q, psi_powers=psi_powers, twiddles=twiddles)
    )

    t0 = time.perf_counter()
    y = fn(x)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t0

    for _ in range(max(0, warmup - 1)):
        y = fn(x)
        jax.block_until_ready(y)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times)
    median_s = float(np.median(arr))
    throughput = (batch * N) / median_s / 1e6
    return compile_s, median_s, throughput


# ── Data collection ──────────────────────────────────────────────────────────

def collect_data():
    rng = np.random.default_rng(SEED)
    device = jax.devices()[0]
    print(f"Device: {device.platform} ({device.device_kind})")

    results = {}
    for logn in LOGN_RANGE:
        N = 1 << logn
        q = provided.generate_ntt_modulus(N, bit_length=BIT_LENGTH)
        psi = provided.negacyclic_psi(N, q)
        for batch in BATCH_RANGE:
            tag = (logn, batch)
            print(f"  logn={logn:2d}  batch={batch:3d} ...", end="", flush=True)
            compile_s, median_s, throughput = bench(
                N, batch, q, psi, RUNS, WARMUP, rng
            )
            results[tag] = {
                "compile_ms": compile_s * 1e3,
                "median_ms": median_s * 1e3,
                "throughput": throughput,
            }
            print(
                f"  compile={compile_s*1e3:8.1f}ms  "
                f"median={median_s*1e3:7.3f}ms  "
                f"{throughput:8.2f} Mcoeff/s"
            )
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_all(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # ── Figure 1: Throughput heatmap ─────────────────────────────────────
    tp = np.full((len(BATCH_RANGE), len(LOGN_RANGE)), np.nan)
    for i, batch in enumerate(BATCH_RANGE):
        for j, logn in enumerate(LOGN_RANGE):
            r = results.get((logn, batch))
            if r:
                tp[i, j] = r["throughput"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        tp,
        aspect="auto",
        origin="lower",
        norm=LogNorm(vmin=max(np.nanmin(tp), 0.1), vmax=np.nanmax(tp)),
        cmap="viridis",
    )
    ax.set_xticks(range(len(LOGN_RANGE)))
    ax.set_xticklabels([str(l) for l in LOGN_RANGE])
    ax.set_yticks(range(len(BATCH_RANGE)))
    ax.set_yticklabels([str(b) for b in BATCH_RANGE])
    ax.set_xlabel("log$_2$(N)")
    ax.set_ylabel("Batch size")
    ax.set_title("NTT Throughput (Mcoeff/s)")
    for i in range(len(BATCH_RANGE)):
        for j in range(len(LOGN_RANGE)):
            if not np.isnan(tp[i, j]):
                ax.text(
                    j, i, f"{tp[i,j]:.1f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if tp[i, j] < np.nanmedian(tp) else "black",
                )
    fig.colorbar(im, ax=ax, label="Mcoeff/s")
    fig.tight_layout()
    fig.savefig("fig_throughput_heatmap.png", dpi=180)
    plt.close(fig)
    print("Saved fig_throughput_heatmap.png")

    # ── Figure 2: Throughput sweep over logn ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch in BATCH_RANGE:
        xs, ys = [], []
        for logn in LOGN_RANGE:
            r = results.get((logn, batch))
            if r:
                xs.append(logn)
                ys.append(r["throughput"])
        ax.plot(xs, ys, "o-", label=f"batch={batch}")
    ax.set_xlabel("log$_2$(N)")
    ax.set_ylabel("Throughput (Mcoeff/s)")
    ax.set_title("NTT Throughput vs Transform Size")
    ax.legend(title="Batch size", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_throughput_sweep.png", dpi=180)
    plt.close(fig)
    print("Saved fig_throughput_sweep.png")

    # ── Figure 3: Compile time + median latency ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for batch in BATCH_RANGE:
        xs, compile_ys, latency_ys = [], [], []
        for logn in LOGN_RANGE:
            r = results.get((logn, batch))
            if r:
                xs.append(logn)
                compile_ys.append(r["compile_ms"])
                latency_ys.append(r["median_ms"])
        ax1.plot(xs, compile_ys, "o-", label=f"batch={batch}")
        ax2.plot(xs, latency_ys, "o-", label=f"batch={batch}")

    ax1.set_xlabel("log$_2$(N)")
    ax1.set_ylabel("Compile time (ms)")
    ax1.set_title("JIT Compile Time")
    ax1.legend(title="Batch size", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("log$_2$(N)")
    ax2.set_ylabel("Median latency (ms)")
    ax2.set_title("Median Execution Latency")
    ax2.legend(title="Batch size", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig("fig_compile_and_latency.png", dpi=180)
    plt.close(fig)
    print("Saved fig_compile_and_latency.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = collect_data()
    try:
        plot_all(results)
    except ImportError:
        print(
            "\nmatplotlib not found. Install it to generate plots:\n"
            "  uv pip install matplotlib"
        )