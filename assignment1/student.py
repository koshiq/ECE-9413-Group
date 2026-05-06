"""
Negacyclic Number Theoretic Transform (NTT) implementation.

GPU-optimized using reshape-based GS-DIF butterflies with Montgomery
multiplication. Each stage reshapes (batch, N) -> (batch, groups, 2, half)
for contiguous memory access — no gather/scatter index arrays needed.
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
    """Return (a + b) mod q, elementwise. Inputs < q < 2^31."""
    result = a + b
    return jnp.where(result >= q, result - q, result)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise. Inputs < q < 2^31."""
    return jnp.where(a >= b, a - b, a + q - b)


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    return (a.astype(jnp.uint64) * b.astype(jnp.uint64) % q).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Montgomery helpers
# -----------------------------------------------------------------------------

def _precompute_q_inv(q_int):
    """Compute -q^-1 mod 2^32 for Montgomery reduction."""
    q_inv = 1
    for _ in range(5):
        q_inv = (q_inv * (2 - q_int * q_inv)) & 0xFFFFFFFF
    return (-q_inv) & 0xFFFFFFFF


# -----------------------------------------------------------------------------
# Table preparation
# -----------------------------------------------------------------------------

def prepare_tables(*, q, psi_powers, twiddles):
    """
    One-time precomputation: per-stage twiddles (Montgomery-scaled),
    bit-reversal permutation, and Montgomery constants.
    """
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

    # Pre-multiply psi_powers by R for Montgomery domain
    psi_powers_mont = (psi_powers.astype(jnp.uint64) * R % q_int).astype(
        jnp.uint32
    )

    # Per-stage twiddles, each pre-multiplied by R for Montgomery
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

    # Bit-reversal permutation
    bit_rev = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rev, val = 0, i
        for _ in range(log2N):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        bit_rev[i] = rev
    bit_rev = jnp.array(bit_rev, dtype=jnp.int32)

    _CACHED_TABLES = {
        "psi_powers_mont": psi_powers_mont,
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
    """
    Forward negacyclic NTT using Gentleman-Sande (DIF).
    Reshape-based butterflies with Montgomery multiplication.
    """
    tables = _CACHED_TABLES
    N = tables["N"]
    log2N = tables["log2N"]
    q_int = tables["q_int"]
    q_inv_int = tables["q_inv_int"]
    psi_mont = tables["psi_powers_mont"]
    stage_twiddles = tables["stage_twiddles"]
    bit_rev = tables["bit_rev"]

    q_u64 = jnp.uint64(q_int)
    q_u32 = jnp.uint32(q_int)
    q_inv_u32 = jnp.uint32(q_inv_int)

    batch = x.shape[0]

    # Step 1: Negacyclic twist — x[n] *= psi^n via Montgomery mul
    z = x.astype(jnp.uint64) * psi_mont.astype(jnp.uint64)
    m = z.astype(jnp.uint32) * q_inv_u32
    t = (z + m.astype(jnp.uint64) * q_u64) >> 32
    t = t.astype(jnp.uint32)
    x = jnp.where(t >= q_u32, t - q_u32, t)

    # Step 2: GS-DIF butterfly stages via reshape
    for stage_idx in range(log2N):
        s = log2N - stage_idx
        m_val = 1 << s
        half_m = m_val >> 1
        num_groups = N // m_val

        # Reshape so butterfly pairs are adjacent on axis 2
        x = x.reshape(batch, num_groups, 2, half_m)
        u = x[:, :, 0, :]  # first half of each group
        v = x[:, :, 1, :]  # second half of each group

        # mod_add: new_even = (u + v) mod q
        s_val = u + v
        new_even = jnp.where(s_val >= q_u32, s_val - q_u32, s_val)

        # mod_sub: diff = (u - v) mod q
        diff = jnp.where(u >= v, u - v, u + q_u32 - v)

        # Montgomery mul: new_odd = diff * twiddle * R^-1 mod q
        tw = stage_twiddles[stage_idx]  # shape (half_m,), broadcasts
        z2 = diff.astype(jnp.uint64) * tw.astype(jnp.uint64)
        m2 = z2.astype(jnp.uint32) * q_inv_u32
        t2 = (z2 + m2.astype(jnp.uint64) * q_u64) >> 32
        t2 = t2.astype(jnp.uint32)
        new_odd = jnp.where(t2 >= q_u32, t2 - q_u32, t2)

        x = jnp.stack([new_even, new_odd], axis=2).reshape(batch, N)

    # Step 3: Bit-reversal permutation
    x = x[:, bit_rev]

    return x