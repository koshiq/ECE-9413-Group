"""
Extra credit: Polynomials from the zkSpeed paper (ISCA '25).

The zkSpeed paper (arXiv:2504.06211) accelerates HyperPlonk's SumCheck
protocol. HyperPlonk uses three main SumCheck instances:

1. ZeroCheck (degree 4): Verifies gate constraints.
   f_zero = q_L*w1*fz + q_R*w2*fz + q_M*w1*w2*fz - q_O*w3*fz + q_C*fz
   Simplified to our expression format (degree-4 terms):
     q_L*w1*fz  (degree 3, 3 vars multiplied)
     q_M*w1*w2*fz (degree 4, 4 vars multiplied)

2. PermCheck (degree 4): Verifies wire permutations.
   Contains terms like p1*p2*fz (degree 3) and D1*D2*D3*fz (degree 4)

3. OpenCheck (degree 2): Verifies polynomial openings.
   Contains terms like y1*k1 (degree 2)

We model these as expressions in the assignment's format and benchmark them
at scales comparable to the paper (num_vars = 20-24, i.e. 2^20 to 2^24 gates).

Usage:
    uv run python extra_credit_zkspeed.py [--num-vars 16 20] [--runs 8] [--warmup 3]
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

ZKSPEED_EXPRESSIONS = {
    "ZeroCheck (deg4)": {
        "expression": [
            ["a", "b", "c"],        # q_L * w1 * fz
            ["a", "d", "c"],        # q_R * w2 * fz
            ["a", "b", "d", "c"],   # q_M * w1 * w2 * fz (degree 4)
            ["a", "e", "c"],        # q_O * w3 * fz
            ["a", "c"],             # q_C * fz
        ],
        "degree": 4,
        "description": "HyperPlonk gate constraint check",
    },
    "PermCheck (deg4)": {
        "expression": [
            ["a", "b", "c", "d"],   # pi * fz2 (degree 4 with accumulator)
            ["a", "b", "c"],        # p1 * p2 * fz2
            ["d", "e", "g", "a"],   # D1 * D2 * D3 * fz2
        ],
        "degree": 4,
        "description": "HyperPlonk wire permutation check",
    },
    "OpenCheck (deg2)": {
        "expression": [
            ["a", "b"],    # y1 * k1
            ["c", "d"],    # y2 * k2
            ["e", "g"],    # y3 * k3
        ],
        "degree": 2,
        "description": "HyperPlonk polynomial opening check",
    },
    "Full HyperPlonk": {
        "expression": [
            ["a", "b", "c", "d"],   # degree-4 term from ZeroCheck
            ["a", "b", "c"],        # degree-3 term
            ["a", "b"],             # degree-2 term
            ["e", "g", "c"],        # additional cross terms
        ],
        "degree": 4,
        "description": "Combined HyperPlonk-style expression",
    },
}


def generate_test_data(num_vars, q_int):
    rng = np.random.default_rng(42)
    N = 1 << num_vars
    tables = {}
    for name in provided.VARIABLE_NAMES:
        tables[name] = rng.integers(0, q_int, size=N, dtype=np.int64)
    return tables


def bench_expression(num_vars, expression, q_int, runs=8, warmup=3):
    tables = generate_test_data(num_vars, q_int)
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
    return compile_s, median_s, p90_s, throughput


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="zkSpeed polynomial benchmark"
    )
    parser.add_argument("--num-vars", type=int, nargs="+", default=[16, 20])
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = jax.devices()[0]
    print(f"Device: {device.platform} ({device.device_kind})")
    print(f"\nzkSpeed / HyperPlonk Polynomial Benchmarks")
    print(f"{'='*90}")

    q_int = 3603169181

    print(f"\n{'expression':<22} {'deg':>3} {'vars':>4} {'N':>10} "
          f"{'compile(ms)':>11} {'median(ms)':>10} {'p90(ms)':>8} {'Mpts/s':>8}")
    print("-" * 90)

    for num_vars in args.num_vars:
        N = 1 << num_vars
        for name, spec in ZKSPEED_EXPRESSIONS.items():
            expression = spec["expression"]
            degree = spec["degree"]

            compile_s, median_s, p90_s, throughput = bench_expression(
                num_vars, expression, q_int,
                runs=args.runs, warmup=args.warmup
            )

            print(f"{name:<22} {degree:>3} {num_vars:>4} {N:>10} "
                  f"{compile_s*1e3:>11.2f} {median_s*1e3:>10.3f} "
                  f"{p90_s*1e3:>8.3f} {throughput:>8.2f}")
        print()

    print("\nComparison with base assignment expressions:")
    print("-" * 90)
    base_expressions = provided.EXPRESSIONS[:4]
    for num_vars in args.num_vars:
        N = 1 << num_vars
        for expr in base_expressions:
            expr_id = provided._expression_id(expr)
            degree = max(len(term) for term in expr)

            compile_s, median_s, p90_s, throughput = bench_expression(
                num_vars, expr, q_int,
                runs=args.runs, warmup=args.warmup
            )

            print(f"{expr_id:<22} {degree:>3} {num_vars:>4} {N:>10} "
                  f"{compile_s*1e3:>11.2f} {median_s*1e3:>10.3f} "
                  f"{p90_s*1e3:>8.3f} {throughput:>8.2f}")
        print()


if __name__ == "__main__":
    main()