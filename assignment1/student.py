"""
Negacyclic Number Theoretic Transform (NTT) implementation.

GPU-optimized using reshape-based GS-DIF butterflies with Montgomery
multiplication. The negacyclic twist is merged with the first butterfly
stage to eliminate one data pass. The last stage skips the twiddle
multiply since MonMul(diff, R) = diff (identity in Montgomery domain).
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

    psi_powers_mont = (psi_powers.astype(jnp.uint64) * R % q_int).astype(
        jnp.uint32
    )

    half = N >> 1
    psi_lo = psi_powers_mont[:half]
    psi_hi = psi_powers_mont[half:]

    stage_twiddles = []
    for stage_idx in range(log2N):
        s = log2N - stage_idx
        m = 1 << s
        half_m = m >> 1
        omega_m = pow(omega, N // m, q_int)

        tw = np.empty(half_m, dtype=np.uint64)
        w = 1
        for j in range(half_m):
            tw[j] = (w * R) % q_int
            w = (w * omega_m) % q_int
        stage_twiddles.append(jnp.array(tw, dtype=jnp.uint32))

    bit_rev = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rev, val = 0, i
        for _ in range(log2N):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        bit_rev[i] = rev
    bit_rev = jnp.array(bit_rev, dtype=jnp.int32)

    _CACHED_TABLES = {
        "psi_lo": psi_lo,
        "psi_hi": psi_hi,
        "stage_twiddles": stage_twiddles,
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
    psi_lo = tables["psi_lo"]
    psi_hi = tables["psi_hi"]
    stage_twiddles = tables["stage_twiddles"]
    bit_rev = tables["bit_rev"]

    q_u64 = jnp.uint64(q_int)
    q_u32 = jnp.uint32(q_int)
    q_inv_u32 = jnp.uint32(q_inv_int)

    batch = x.shape[0]
    half = N >> 1

    # Stage 0: merged negacyclic twist + first GS-DIF butterfly
    u_raw = x[:, :half]
    v_raw = x[:, half:]

    z = u_raw.astype(jnp.uint64) * psi_lo.astype(jnp.uint64)
    m = z.astype(jnp.uint32) * q_inv_u32
    t = (z + m.astype(jnp.uint64) * q_u64) >> 32
    u = t.astype(jnp.uint32)
    u = jnp.where(u >= q_u32, u - q_u32, u)

    z = v_raw.astype(jnp.uint64) * psi_hi.astype(jnp.uint64)
    m = z.astype(jnp.uint32) * q_inv_u32
    t = (z + m.astype(jnp.uint64) * q_u64) >> 32
    v = t.astype(jnp.uint32)
    v = jnp.where(v >= q_u32, v - q_u32, v)

    s_val = u + v
    new_even = jnp.where(s_val >= q_u32, s_val - q_u32, s_val)
    diff = jnp.where(u >= v, u - v, u + q_u32 - v)

    tw = stage_twiddles[0]
    z = diff.astype(jnp.uint64) * tw.astype(jnp.uint64)
    m = z.astype(jnp.uint32) * q_inv_u32
    t = (z + m.astype(jnp.uint64) * q_u64) >> 32
    t = t.astype(jnp.uint32)
    new_odd = jnp.where(t >= q_u32, t - q_u32, t)

    x = jnp.stack([new_even, new_odd], axis=1).reshape(batch, N)

    # Stages 1 through log2N-1
    for stage_idx in range(1, log2N):
        s = log2N - stage_idx
        m_val = 1 << s
        half_m = m_val >> 1
        num_groups = N // m_val

        x = x.reshape(batch, num_groups, 2, half_m)
        u = x[:, :, 0, :]
        v = x[:, :, 1, :]

        s_val = u + v
        new_even = jnp.where(s_val >= q_u32, s_val - q_u32, s_val)
        diff = jnp.where(u >= v, u - v, u + q_u32 - v)

        if half_m == 1:
            new_odd = diff
        else:
            tw = stage_twiddles[stage_idx]
            z = diff.astype(jnp.uint64) * tw.astype(jnp.uint64)
            m = z.astype(jnp.uint32) * q_inv_u32
            t = (z + m.astype(jnp.uint64) * q_u64) >> 32
            t = t.astype(jnp.uint32)
            new_odd = jnp.where(t >= q_u32, t - q_u32, t)

        x = jnp.stack([new_even, new_odd], axis=2).reshape(batch, N)

    x = x[:, bit_rev]
    return x