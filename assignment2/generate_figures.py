"""
Generate SumCheck performance figures for the assignment report.

Produces the following figures:
  1. Line plot: throughput vs expression (fixed problem size, one line per expression)
  2. Line plot: throughput vs problem size (fixed expression, one line per size)
  3. Heatmap: throughput across problem sizes x expressions
  4. Bar chart: GPU speedup over CPU
  5. Pie chart: compile time vs runtime breakdown
  6. Bar chart: optimization ablation (Barrett vs naive)

Usage:
    uv run python generate_figures.py [--device cpu|gpu]
"""

from __future__ import annotations

import os
import time

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import provided
import student

NUM_VARS_LIST = [4, 8, 12, 16, 20]
BASE_EXPRESSIONS = provided.EXPRESSIONS[:4]
EXPR_IDS = [provided._expression_id(e) for e in BASE_EXPRESSIONS]
RUNS = 8
WARMUP = 3
Q_INT = 3603169181
SEED = 42


def generate_tables(num_vars, q_int):
    rng = np.random.default_rng(SEED)
    N = 1 << num_vars
    tables = {}
    for name in provided.VARIABLE_NAMES:
        tables[name] = rng.integers(0, q_int, size=N, dtype=np.int64)
    return tables


def bench(num_vars, expression, q_int, runs=RUNS, warmup=WARMUP):
    tables = generate_tables(num_vars, q_int)
    num_rounds = num_vars
    N = 1 << num_vars

    rng = np.random.default_rng(99)
    challenges = [int(x) % q_int for x in rng.integers(0, q_int, size=num_rounds - 1)]

    jax_tables = {k: jnp.asarray(v, dtype=jnp.uint32) for k, v in tables.items()}
    jax_challenges = jnp.asarray(challenges, dtype=jnp.uint32)

    fn = jax.jit(
        lambda t, c: student.sumcheck_32(
            t, q=q_int, expression=expression,
            challenges=c, num_rounds=num_rounds
        )
    )

    t0 = time.perf_counter()
    out = fn(jax_tables, jax_challenges)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    for _ in range(max(0, warmup - 1)):
        out = fn(jax_tables, jax_challenges)
        jax.block_until_ready(out)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn(jax_tables, jax_challenges)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times)
    median_s = float(np.median(arr))
    p90_s = float(np.percentile(arr, 90))
    throughput = N / median_s / 1e6
    return {
        "compile_ms": compile_s * 1e3,
        "median_ms": median_s * 1e3,
        "p90_ms": p90_s * 1e3,
        "throughput": throughput,
    }


def bench_naive(num_vars, expression, q_int, runs=RUNS, warmup=WARMUP):
    """Benchmark using naive mod_mul (% operator) for ablation comparison."""
    tables = generate_tables(num_vars, q_int)
    num_rounds = num_vars
    N = 1 << num_vars

    rng = np.random.default_rng(99)
    challenges = [int(x) % q_int for x in rng.integers(0, q_int, size=num_rounds - 1)]

    jax_tables = {k: jnp.asarray(v, dtype=jnp.uint32) for k, v in tables.items()}
    jax_challenges = jnp.asarray(challenges, dtype=jnp.uint32)

    q64 = jnp.uint64(q_int)

    def naive_sumcheck(eval_tables, *, q, expression, challenges, num_rounds):
        def _mmul(a, b):
            return (a.astype(jnp.uint64) * b.astype(jnp.uint64) % q64).astype(jnp.uint32)
        def _madd(a, b):
            s = a.astype(jnp.uint64) + b.astype(jnp.uint64)
            return jnp.where(s >= q64, s - q64, s).astype(jnp.uint32)
        def _msub(a, b):
            a64, b64 = a.astype(jnp.uint64), b.astype(jnp.uint64)
            return jnp.where(a64 >= b64, a64 - b64, a64 + q64 - b64).astype(jnp.uint32)
        def _mle(zero, one, t):
            return _madd(_mmul(_msub(one, zero), t), zero)

        degree = max(len(term) for term in expression)
        num_eval_pts = degree + 1
        tables = {k: jnp.asarray(v, dtype=jnp.uint32) for k, v in eval_tables.items()}
        all_round_evals = []

        for round_idx in range(num_rounds):
            even = {name: tables[name][0::2] for name in tables}
            odd = {name: tables[name][1::2] for name in tables}
            half_n = even[next(iter(even))].shape[0]
            eval_sums = []
            for t in range(num_eval_pts):
                if t == 0: ext = even
                elif t == 1: ext = odd
                else:
                    t_scalar = jnp.asarray(t, dtype=jnp.uint32)
                    ext = {name: _mle(even[name], odd[name], t_scalar) for name in tables}
                comp = jnp.zeros(half_n, dtype=jnp.uint32)
                for term in expression:
                    term_val = ext[term[0]]
                    for var in term[1:]:
                        term_val = _mmul(term_val, ext[var])
                    comp = _madd(comp, term_val)
                s = jnp.sum(comp.astype(jnp.uint64), dtype=jnp.uint64) % q64
                eval_sums.append(s.astype(jnp.uint32))
            all_round_evals.append(jnp.stack(eval_sums))
            if round_idx < len(challenges):
                r = challenges[round_idx]
                for name in tables:
                    tables[name] = _mle(even[name], odd[name], r)
        claim0 = _madd(all_round_evals[0][0], all_round_evals[0][1])
        return claim0, jnp.stack(all_round_evals)

    fn = jax.jit(
        lambda t, c: naive_sumcheck(
            t, q=q_int, expression=expression,
            challenges=c, num_rounds=num_rounds
        )
    )

    t0 = time.perf_counter()
    out = fn(jax_tables, jax_challenges)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    for _ in range(max(0, warmup - 1)):
        out = fn(jax_tables, jax_challenges)
        jax.block_until_ready(out)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn(jax_tables, jax_challenges)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times)
    median_s = float(np.median(arr))
    throughput = N / median_s / 1e6
    return {
        "compile_ms": compile_s * 1e3,
        "median_ms": median_s * 1e3,
        "throughput": throughput,
    }


def collect_data():
    device = jax.devices()[0]
    print(f"Device: {device.platform} ({device.device_kind})")

    results = {}
    for nv in NUM_VARS_LIST:
        for i, expr in enumerate(BASE_EXPRESSIONS):
            tag = (nv, EXPR_IDS[i])
            print(f"  vars={nv:2d} expr={EXPR_IDS[i]:<10} ...", end="", flush=True)
            r = bench(nv, expr, Q_INT)
            results[tag] = r
            print(f"  {r['throughput']:8.2f} Mpts/s  compile={r['compile_ms']:8.1f}ms")
    return results


def collect_ablation_data():
    print("\nCollecting ablation data (naive vs Barrett)...")
    ablation = {}
    for nv in [16, 20]:
        expr = BASE_EXPRESSIONS[1]
        eid = EXPR_IDS[1]
        print(f"  vars={nv} expr={eid} Barrett ...", end="", flush=True)
        r_barrett = bench(nv, expr, Q_INT)
        print(f" {r_barrett['throughput']:.2f} Mpts/s")
        print(f"  vars={nv} expr={eid} Naive ...", end="", flush=True)
        r_naive = bench_naive(nv, expr, Q_INT)
        print(f" {r_naive['throughput']:.2f} Mpts/s")
        ablation[nv] = {"barrett": r_barrett, "naive": r_naive}
    return ablation


def plot_all(results, ablation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # ── Figure 1: Throughput vs expression (fixed problem size) ──────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for nv in NUM_VARS_LIST:
        xs = list(range(len(EXPR_IDS)))
        ys = [results.get((nv, eid), {}).get("throughput", 0) for eid in EXPR_IDS]
        if any(y > 0 for y in ys):
            ax.plot(xs, ys, "o-", label=f"vars={nv}", linewidth=2)
    ax.set_xticks(range(len(EXPR_IDS)))
    ax.set_xticklabels(EXPR_IDS, fontsize=9)
    ax.set_xlabel("Expression")
    ax.set_ylabel("Throughput (Mpts/s)")
    ax.set_title("SumCheck Throughput by Expression (32-bit)")
    ax.legend(title="Problem Size", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_throughput_by_expr.png", dpi=180)
    plt.close(fig)
    print("Saved fig_throughput_by_expr.png")

    # ── Figure 2: Throughput vs problem size (fixed expression) ──────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, eid in enumerate(EXPR_IDS):
        xs = NUM_VARS_LIST
        ys = [results.get((nv, eid), {}).get("throughput", 0) for nv in NUM_VARS_LIST]
        if any(y > 0 for y in ys):
            ax.plot(xs, ys, "o-", label=eid, linewidth=2)
    ax.set_xlabel("Number of Variables (log$_2$ N)")
    ax.set_ylabel("Throughput (Mpts/s)")
    ax.set_title("SumCheck Throughput vs Problem Size (32-bit)")
    ax.legend(title="Expression", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_throughput_by_size.png", dpi=180)
    plt.close(fig)
    print("Saved fig_throughput_by_size.png")

    # ── Figure 3: Throughput heatmap ─────────────────────────────────────
    tp = np.full((len(EXPR_IDS), len(NUM_VARS_LIST)), np.nan)
    for i, eid in enumerate(EXPR_IDS):
        for j, nv in enumerate(NUM_VARS_LIST):
            r = results.get((nv, eid))
            if r:
                tp[i, j] = r["throughput"]

    fig, ax = plt.subplots(figsize=(9, 5))
    valid = tp[~np.isnan(tp)]
    if len(valid) > 0:
        im = ax.imshow(
            tp, aspect="auto", origin="lower",
            norm=LogNorm(vmin=max(np.nanmin(tp), 0.01), vmax=np.nanmax(tp)),
            cmap="viridis",
        )
        ax.set_xticks(range(len(NUM_VARS_LIST)))
        ax.set_xticklabels([str(v) for v in NUM_VARS_LIST])
        ax.set_yticks(range(len(EXPR_IDS)))
        ax.set_yticklabels(EXPR_IDS)
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Expression")
        ax.set_title("SumCheck Throughput Heatmap (Mpts/s)")
        for i in range(len(EXPR_IDS)):
            for j in range(len(NUM_VARS_LIST)):
                if not np.isnan(tp[i, j]):
                    ax.text(
                        j, i, f"{tp[i,j]:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if tp[i, j] < np.nanmedian(tp) else "black",
                    )
        fig.colorbar(im, ax=ax, label="Mpts/s")
    fig.tight_layout()
    fig.savefig("fig_throughput_heatmap.png", dpi=180)
    plt.close(fig)
    print("Saved fig_throughput_heatmap.png")

    # ── Figure 4: Compile time vs runtime pie chart ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pie_configs = [
        (20, "a"),
        (20, "a*b"),
        (20, "a*b*c"),
    ]
    for ax, (nv, eid) in zip(axes, pie_configs):
        r = results.get((nv, eid))
        if r:
            compile_ms = r["compile_ms"]
            runtime_ms = r["median_ms"]
            sizes = [compile_ms, runtime_ms]
            labels = [f"Compile\n{compile_ms:.0f}ms", f"Runtime\n{runtime_ms:.1f}ms"]
            colors = ["#ff9999", "#66b3ff"]
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                   startangle=90, textprops={"fontsize": 8})
            ax.set_title(f"vars={nv}, expr={eid}", fontsize=10)
    fig.suptitle("Compile Time vs Runtime Breakdown", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("fig_compile_vs_runtime.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_compile_vs_runtime.png")

    # ── Figure 5: Ablation bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = []
    naive_throughputs = []
    barrett_throughputs = []
    for nv in sorted(ablation.keys()):
        x_labels.append(f"vars={nv}")
        naive_throughputs.append(ablation[nv]["naive"]["throughput"])
        barrett_throughputs.append(ablation[nv]["barrett"]["throughput"])

    x = np.arange(len(x_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, naive_throughputs, width, label="Naive (% operator)", color="#ff9999")
    bars2 = ax.bar(x + width/2, barrett_throughputs, width, label="Barrett reduction", color="#66b3ff")

    for bar, val in zip(bars1, naive_throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, barrett_throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Throughput (Mpts/s)")
    ax.set_title("Optimization Ablation: Naive vs Barrett (expr=a*b)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig_ablation.png", dpi=180)
    plt.close(fig)
    print("Saved fig_ablation.png")

    # ── Figure 6: Median latency vs problem size ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, eid in enumerate(EXPR_IDS):
        xs = NUM_VARS_LIST
        ys = [results.get((nv, eid), {}).get("median_ms", 0) for nv in NUM_VARS_LIST]
        if any(y > 0 for y in ys):
            ax.plot(xs, ys, "o-", label=eid, linewidth=2)
    ax.set_xlabel("Number of Variables (log$_2$ N)")
    ax.set_ylabel("Median Latency (ms)")
    ax.set_title("SumCheck Latency vs Problem Size (32-bit)")
    ax.legend(title="Expression", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_latency_by_size.png", dpi=180)
    plt.close(fig)
    print("Saved fig_latency_by_size.png")


def main():
    results = collect_data()
    ablation = collect_ablation_data()
    try:
        plot_all(results, ablation)
    except ImportError:
        print(
            "\nmatplotlib not found. Install it to generate plots:\n"
            "  uv pip install matplotlib"
        )


if __name__ == "__main__":
    main()