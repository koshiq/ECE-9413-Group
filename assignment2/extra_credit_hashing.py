"""
Extra credit: Fiat-Shamir hashing between SumCheck rounds.

In the interactive SumCheck protocol, the verifier sends random challenges.
In a non-interactive (Fiat-Shamir) version, the prover derives each challenge
by hashing the transcript so far (claim0 + all round evaluations emitted).

This benchmark measures the overhead of Fiat-Shamir hashing vs. the baseline
SumCheck with pre-supplied challenges.

Usage:
    uv run python fiat_shamir_bench.py [--num-vars 4|16|20] [--device cpu|gpu]
"""

from __future__ import annotations

import hashlib
import os
import struct
import time

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import provided
import student


def sha256_challenge(transcript_bytes: bytes, q: int) -> int:
    h = hashlib.sha256(transcript_bytes).digest()
    val = int.from_bytes(h[:8], "little")
    return val % q


def evals_to_bytes(evals: list[int]) -> bytes:
    return b"".join(struct.pack("<I", int(v)) for v in evals)


def sumcheck_fiat_shamir_32(eval_tables, *, q, expression, num_rounds):
    """32-bit SumCheck with Fiat-Shamir challenge derivation.

    Instead of receiving challenges from a verifier, the prover hashes the
    round evaluations to derive the next challenge. This is the standard
    Fiat-Shamir transform applied to the SumCheck protocol.

    Note: This cannot be fully JIT'd because the hash is computed in Python.
    Each round requires a device-to-host transfer of the round evaluations,
    a Python SHA-256 computation, and a host-to-device transfer of the challenge.
    """
    q_int = int(q)
    q64 = jnp.uint64(q_int)

    _r16 = jnp.uint64(pow(2, 16, q_int))
    _r32 = jnp.uint64(pow(2, 32, q_int))
    _r48 = jnp.uint64((pow(2, 16, q_int) * pow(2, 32, q_int)) % q_int)
    _inv_q = jnp.uint64((1 << 63) // q_int)
    _M16 = jnp.uint64(0xFFFF)

    def _reduce(z):
        s = (z & _M16) + ((z >> 16) & _M16) * _r16 + \
            ((z >> 32) & _M16) * _r32 + (z >> 48) * _r48
        approx = ((s >> 31).astype(jnp.uint32).astype(jnp.uint64) * _inv_q) >> 32
        r = s - approx * q64
        return jnp.where(r >= q64, r - q64, r).astype(jnp.uint32)

    def _mmul(a, b):
        return _reduce(a.astype(jnp.uint64) * b.astype(jnp.uint64))

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
    transcript = struct.pack("<I", q_int)

    for round_idx in range(num_rounds):
        even = {name: tables[name][0::2] for name in tables}
        odd = {name: tables[name][1::2] for name in tables}
        half_n = even[next(iter(even))].shape[0]

        eval_sums = []
        for t in range(num_eval_pts):
            if t == 0:
                ext = even
            elif t == 1:
                ext = odd
            else:
                t_scalar = jnp.asarray(t, dtype=jnp.uint32)
                ext = {
                    name: _mle(even[name], odd[name], t_scalar)
                    for name in tables
                }

            comp = jnp.zeros(half_n, dtype=jnp.uint32)
            for term in expression:
                term_val = ext[term[0]]
                for var in term[1:]:
                    term_val = _mmul(term_val, ext[var])
                comp = _madd(comp, term_val)

            eval_sums.append(
                _reduce(jnp.sum(comp.astype(jnp.uint64), dtype=jnp.uint64))
            )

        round_evals = jnp.stack(eval_sums)
        all_round_evals.append(round_evals)

        jax.block_until_ready(round_evals)
        evals_host = [int(v) for v in round_evals]
        transcript += evals_to_bytes(evals_host)

        if round_idx < num_rounds - 1:
            r_int = sha256_challenge(transcript, q_int)
            r = jnp.asarray(r_int, dtype=jnp.uint32)
            for name in tables:
                tables[name] = _mle(even[name], odd[name], r)

    claim0 = _madd(all_round_evals[0][0], all_round_evals[0][1])
    round_evals_array = jnp.stack(all_round_evals)
    return claim0, round_evals_array


def generate_test_data(num_vars, q_int):
    rng = np.random.default_rng(42)
    N = 1 << num_vars
    var_names = list(provided.VARIABLE_NAMES)
    tables = {}
    for name in var_names:
        tables[name] = rng.integers(0, q_int, size=N, dtype=np.int64)
    return tables


def bench_one(num_vars, expression, q_int, mode, runs=8, warmup=3):
    tables = generate_test_data(num_vars, q_int)
    num_rounds = num_vars
    N = 1 << num_vars

    jax_tables = {k: jnp.asarray(v, dtype=jnp.uint32) for k, v in tables.items()}

    if mode == "baseline":
        rng = np.random.default_rng(99)
        challenges = [int(x) % q_int for x in rng.integers(0, q_int, size=num_rounds - 1)]
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

        for _ in range(warmup - 1):
            out = fn(jax_tables, jax_challenges)
            jax.block_until_ready(out)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out = fn(jax_tables, jax_challenges)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

    else:
        t0 = time.perf_counter()
        out = sumcheck_fiat_shamir_32(
            jax_tables, q=q_int, expression=expression, num_rounds=num_rounds
        )
        jax.block_until_ready(out)
        compile_s = time.perf_counter() - t0

        for _ in range(warmup - 1):
            out = sumcheck_fiat_shamir_32(
                jax_tables, q=q_int, expression=expression, num_rounds=num_rounds
            )
            jax.block_until_ready(out)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out = sumcheck_fiat_shamir_32(
                jax_tables, q=q_int, expression=expression, num_rounds=num_rounds
            )
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

    arr = np.asarray(times)
    median_s = float(np.median(arr))
    throughput = N / median_s / 1e6
    return compile_s, median_s, throughput


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fiat-Shamir hashing benchmark")
    parser.add_argument("--num-vars", type=int, nargs="+", default=[4, 16, 20])
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = jax.devices()[0]
    print(f"Device: {device.platform} ({device.device_kind})")

    base_expressions = provided.EXPRESSIONS[:4]
    q_int = 3603169181

    print(f"\n{'expr':<12} {'vars':>4} {'mode':<12} "
          f"{'compile(ms)':>11} {'median(ms)':>10} {'Mpts/s':>8} {'slowdown':>8}")
    print("-" * 80)

    for num_vars in args.num_vars:
        for expr in base_expressions:
            expr_id = provided._expression_id(expr)

            _, med_base, tp_base = bench_one(
                num_vars, expr, q_int, "baseline",
                runs=args.runs, warmup=args.warmup
            )
            compile_fs, med_fs, tp_fs = bench_one(
                num_vars, expr, q_int, "fiat_shamir",
                runs=args.runs, warmup=args.warmup
            )

            slowdown = med_fs / med_base if med_base > 0 else float("inf")

            print(f"{expr_id:<12} {num_vars:>4} {'baseline':<12} "
                  f"{'--':>11} {med_base*1e3:>10.3f} {tp_base:>8.2f} {'--':>8}")
            print(f"{expr_id:<12} {num_vars:>4} {'fiat-shamir':<12} "
                  f"{compile_fs*1e3:>11.1f} {med_fs*1e3:>10.3f} {tp_fs:>8.2f} "
                  f"{slowdown:>7.1f}x")


if __name__ == "__main__":
    main()