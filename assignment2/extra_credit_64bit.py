"""
Extra credit: Optimized 64-bit SumCheck path.

Benchmarks the 64-bit sumcheck using Russian-peasant modular multiplication
(JAX has no native uint128) with inlined arithmetic to reduce function call
overhead. Compares against CPU baseline.

Usage:
    uv run python extra_credit_64bit.py [--num-vars 4 16] [--runs 8] [--warmup 3]
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


def generate_64bit_test_data(num_vars, q_int):
    rng = np.random.default_rng(42)
    N = 1 << num_vars
    tables = {}
    for name in provided.VARIABLE_NAMES:
        tables[name] = rng.integers(0, q_int, size=N, dtype=np.uint64)
    return tables


def bench_64bit(num_vars, expression, q_int, runs=8, warmup=3):
    tables = generate_64bit_test_data(num_vars, q_int)
    num_rounds = num_vars

    rng = np.random.default_rng(99)
    challenges = [int(x) % q_int for x in rng.integers(0, q_int, size=num_rounds - 1, dtype=np.uint64)]
    N = 1 << num_vars

    jax_tables = {k: jnp.asarray(v, dtype=jnp.uint64) for k, v in tables.items()}
    jax_challenges = jnp.asarray(challenges, dtype=jnp.uint64)

    fn = jax.jit(
        lambda t, c: student.sumcheck_64(
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
    return compile_s, median_s, p90_s, throughput


def main():
    import argparse
    parser = argparse.ArgumentParser(description="64-bit SumCheck benchmark")
    parser.add_argument("--num-vars", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = jax.devices()[0]
    print(f"Device: {device.platform} ({device.device_kind})")
    print(f"\n64-bit SumCheck Benchmark")
    print(f"{'='*80}")

    q_int = 13419220890677911291

    base_expressions = provided.EXPRESSIONS[:4]

    print(f"\n{'testcase':<20} {'bits':>4} {'N':>10} {'expr':<12} "
          f"{'compile(ms)':>11} {'median(ms)':>10} {'p90(ms)':>8} {'Mpts/s':>8}")
    print("-" * 90)

    for num_vars in args.num_vars:
        N = 1 << num_vars
        for expr in base_expressions:
            expr_id = provided._expression_id(expr)

            compile_s, median_s, p90_s, throughput = bench_64bit(
                num_vars, expr, q_int,
                runs=args.runs, warmup=args.warmup
            )

            case_name = f"v{num_vars}_64bit"
            print(f"{case_name:<20} {64:>4} {N:>10} {expr_id:<12} "
                  f"{compile_s*1e3:>11.2f} {median_s*1e3:>10.3f} "
                  f"{p90_s*1e3:>8.3f} {throughput:>8.2f}")


if __name__ == "__main__":
    main()