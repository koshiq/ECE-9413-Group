"""
Negacyclic Number Theoretic Transform (NTT) implementation.

GS-DIF with reshape-based butterflies and Montgomery multiplication.
Unrolled Python loop gives XLA full visibility for cross-stage fusion.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_CACHED_TABLES = {}


# -----------------------------------------------------------------------------
# Modular Arithmetic (public API)
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    result = a + b
    return jnp.where(result >= q, result - q, result)


def mod_sub(a, b, q):
    return jnp.where(a >= b, a - b, a + q - b)


def mod_mul(a, b, q):
    return (a.astype(jnp.uint64) * b.astype(jnp.uint64) % q).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Montgomery helpers
# -----------------------------------------------------------------------------

def _precompute_q_inv(q_int):
    q_inv = 1
    for _ in range(5):
        q_inv = (q_inv * (2 - q_int * q_inv)) & 0xFFFFFFFF
    return (-q_inv) & 0xFFFFFFFF


# -----------------------------------------------------------------------------
# Table preparation
# -----------------------------------------------------------------------------

def prepare_tables(*, q, psi_powers, twiddles):
    global _CACHED_TABLES

    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles_arr = jnp.asarray(twiddles, dtype=jnp.uint32)

    N = int(psi_powers.shape[0])
    log2N = int(np.log2(N))
    q_int = int(q)

    psi_val = int(psi_powers[1])
    omega = pow(psi_val, 2, q_int)

    R = (1 << 32) % q_int
    q_inv_int = _precompute_q_inv(q_int)

    psi_mont = np.empty(N, dtype=np.uint64)
    cur = 1
    for i in range(N):
        psi_mont[i] = (cur * R) % q_int
        cur = (cur * psi_val) % q_int
    psi_mont = jnp.array(psi_mont, dtype=jnp.uint32)

    tw_stages = []
    for s in range(log2N):
        half_m = N >> (s + 1)
        omega_step = pow(omega, 1 << s, q_int)
        tw = np.empty(half_m, dtype=np.uint64)
        w = 1
        for j in range(half_m):
            tw[j] = (w * R) % q_int
            w = (w * omega_step) % q_int
        tw_stages.append(jnp.array(tw, dtype=jnp.uint32))

    bit_rev = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rev, val = 0, i
        for _ in range(log2N):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        bit_rev[i] = rev
    bit_rev = jnp.array(bit_rev, dtype=jnp.int32)

    _CACHED_TABLES = {
        "psi_mont": psi_mont,
        "tw_stages": tw_stages,
        "bit_rev": bit_rev,
        "N": N,
        "log2N": log2N,
        "q_int": q_int,
        "q_inv_int": q_inv_int,
    }

    return psi_powers, twiddles_arr


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------

@jax.jit
def ntt(x, *, q, psi_powers, twiddles):
    tables = _CACHED_TABLES
    N = tables["N"]
    log2N = tables["log2N"]
    q_int = tables["q_int"]
    q_inv_int = tables["q_inv_int"]
    psi_mont = tables["psi_mont"]
    tw_stages = tables["tw_stages"]
    bit_rev = tables["bit_rev"]

    q64 = jnp.uint64(q_int)
    q32 = jnp.uint32(q_int)
    qi32 = jnp.uint32(q_inv_int)
    B = x.shape[0]

    def mont(a, b):
        z = a.astype(jnp.uint64) * b.astype(jnp.uint64)
        m = z.astype(jnp.uint32) * qi32
        t = (z + m.astype(jnp.uint64) * q64) >> 32
        r = t.astype(jnp.uint32)
        return jnp.where(r >= q32, r - q32, r)

    def madd(a, b):
        s = a + b
        return jnp.where(s >= q32, s - q32, s)

    def msub(a, b):
        return jnp.where(a >= b, a - b, a + q32 - b)

    x = mont(x, psi_mont)

    for s in range(log2N):
        num_groups = 1 << s
        half_m = N >> (s + 1)

        x = x.reshape(B, num_groups, 2, half_m)
        top = x[:, :, 0, :]
        bot = x[:, :, 1, :]

        sum_val = madd(top, bot)
        diff = msub(top, bot)
        prod = mont(diff, tw_stages[s])

        x = jnp.stack([sum_val, prod], axis=2).reshape(B, N)

    x = x[:, bit_rev]
    return x