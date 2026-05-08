"""
Extra credit: 128-bit SumCheck primitives and benchmark.

JAX has no native uint128, so the 128-bit path uses Python big-int arithmetic.
This benchmark measures correctness and performance of 128-bit modular
arithmetic primitives and the full 128-bit SumCheck protocol.

Usage:
    uv run python extra_credit_128bit.py [--num-vars 4 8] [--runs 8]
"""

from __future__ import annotations

import os
import time

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import provided
import student


def generate_128bit_prime():
    """Generate a 128-bit prime for testing."""
    return (1 << 127) - 1  # Mersenne prime M127


def verify_primitives():
    """Verify 128-bit primitive correctness with edge cases."""
    q = generate_128bit_prime()
    print("128-bit Primitive Verification")
    print(f"  q = 2^127 - 1 = {q}")
    print(f"  q bits = {q.bit_length()}")

    test_cases = [
        ("zero+zero", 0, 0),
        ("one+one", 1, 1),
        ("max+max", q - 1, q - 1),
        ("max+1", q - 1, 1),
        ("large*large", q - 2, q - 3),
        ("mid*mid", q // 2, q // 3),
    ]

    all_pass = True
    for name, a, b in test_cases:
        add_result = student.mod_add_128(a, b, q)
        expected_add = (a + b) % q
        sub_result = student.mod_sub_128(a, b, q)
        expected_sub = (a - b) % q
        mul_result = student.mod_mul_128(a, b, q)
        expected_mul = (a * b) % q

        ok_add = add_result == expected_add
        ok_sub = sub_result == expected_sub
        ok_mul = mul_result == expected_mul

        status = "PASS" if (ok_add and ok_sub and ok_mul) else "FAIL"
        if not (ok_add and ok_sub and ok_mul):
            all_pass = False
        print(f"  {name}: add={ok_add} sub={ok_sub} mul={ok_mul} [{status}]")

    mle = student.mle_update_128(10, 20, 5, q=q)
    expected_mle = ((20 - 10) * 5 + 10) % q
    ok_mle = mle == expected_mle
    print(f"  mle_update: {ok_mle} [{'PASS' if ok_mle else 'FAIL'}]")
    all_pass = all_pass and ok_mle

    return all_pass


def generate_128bit_test_data(num_vars, q):
    rng = np.random.default_rng(42)
    N = 1 << num_vars
    tables = {}
    for name in provided.VARIABLE_NAMES:
        vals = [int(rng.integers(0, min(q, 2**63))) for _ in range(N)]
        tables[name] = vals
    return tables


def bench_128bit(num_vars, expression, q, runs=8):
    tables = generate_128bit_test_data(num_vars, q)
    num_rounds = num_vars

    rng = np.random.default_rng(99)
    challenges = [int(rng.integers(0, min(q, 2**63))) % q for _ in range(num_rounds - 1)]
    N = 1 << num_vars

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        claim0, round_evals = student.sumcheck_128(
            tables, q=q, expression=expression,
            challenges=challenges, num_rounds=num_rounds
        )
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times)
    median_s = float(np.median(arr))
    throughput = N / median_s / 1e6 if median_s > 0 else 0
    return median_s, throughput, claim0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="128-bit SumCheck benchmark")
    parser.add_argument("--num-vars", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    all_pass = verify_primitives()
    print(f"\nAll primitives {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    q = generate_128bit_prime()

    base_expressions = provided.EXPRESSIONS[:4]

    print(f"\n128-bit SumCheck Benchmark (pure Python big-int arithmetic)")
    print(f"{'testcase':<16} {'bits':>4} {'N':>8} {'expr':<12} "
          f"{'median(ms)':>10} {'Mpts/s':>8} {'claim0_bits':>11}")
    print("-" * 75)

    for num_vars in args.num_vars:
        N = 1 << num_vars
        for expr in base_expressions:
            expr_id = provided._expression_id(expr)
            median_s, throughput, claim0 = bench_128bit(
                num_vars, expr, q, runs=args.runs
            )
            claim_bits = int(claim0).bit_length() if claim0 else 0
            case_name = f"v{num_vars}_128bit"
            print(f"{case_name:<16} {128:>4} {N:>8} {expr_id:<12} "
                  f"{median_s*1e3:>10.3f} {throughput:>8.4f} {claim_bits:>11}")


if __name__ == "__main__":
    main()