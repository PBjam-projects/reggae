#!/usr/bin/env python3
# scripsit Joel Ong 2025
# jax'ed up versions of functions from implementation
# of asymptotic eigenvalue equation in Ong & Gehan 2023,
# as also implemented for PBJam 2's RGBL1Model.

from functools import partial
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

interps = {
    'np': lambda xi, yi, x: jnp.interp(x, xi, yi),
}

INTERP = 'np'

## DEFINITIONS

# Evaluating the phase functions and ζ

@jax.jit
def nearest(ν, ν_target):
    '''
    Utility function: given 1d arrays ν and ν_target,
    return a 1d array with the same shape as ν, containing
    the nearest elements of ν_target to each element of ν.
    '''
    if len(jnp.array(ν_target).shape) == 0:
        return ν_target * (0 * ν + 1)
    if len(jnp.array(ν).shape) == 0:
        return ν_target[jnp.argmin(jnp.abs(ν_target - ν))]
    return ν_target[jnp.argmin(jnp.abs(ν[:, None] - ν_target[None, :]), axis=1)]

@jax.jit
def Θ_p(ν, Δν, ν_p):
    '''
    p-mode phase function Θ_p. Either provide a list of
    p-mode frequencies ν_p, or a callable phase function ε_p.
    '''
    return jnp.pi * jnp.where(
        (ν <= jnp.max(ν_p)) & (ν >= jnp.min(ν_p)),
        interps[INTERP](ν_p, jnp.arange(len(ν_p)), ν),
        (ν - nearest(ν, ν_p)) / Δν + jnp.round((nearest(ν,ν_p)-ν_p[0]) / Δν)
    )

@jax.jit
def Θ_g(ν, ΔΠ, ν_g):
    '''
    g-mode phase function Θ_g. Either provide a list of
    g-mode frequencies ν_g, or a callable phase function ε_g.
    '''

    return jnp.pi * jnp.where(
        (ν <= jnp.max(ν_g)) & (ν >= jnp.min(ν_g)),
        -interps[INTERP](jnp.sort(1/ν_g), jnp.arange(len(ν_g)), 1/ν),
        (1/nearest(ν, ν_g) - 1/ν) / ΔΠ
    )

@jax.jit
def ζ(ν, q, ΔΠ, Δν, ν_p, ν_g):
    '''
    ζ, the approximate local mixing fraction.
    '''
    θ_p = Θ_p(ν, Δν, ν_p)
    θ_g = Θ_g(ν, ΔΠ, ν_g)
    return 1/(1 + ΔΠ / Δν * ν**2 / q * jnp.sin(θ_g)**2 / jnp.cos(θ_p)**2)

@jax.jit
def ζ_p(ν, q, ΔΠ, Δν, ν_p):
    '''
    ζ as defined using only the p-mode phase function.
    Agrees with ζ only at the eigenvalues (i.e. roots of the characteristic
    equation F(ν) = 0).
    '''
    θ = Θ_p(ν, Δν, ν_p)
    return 1/(1 + ΔΠ / Δν * ν**2 / (q * jnp.cos(θ)**2 + jnp.sin(θ)**2/q))

@jax.jit
def ζ_g(ν, q, ΔΠ, Δν, ν_g):
    '''
    ζ as defined using only the g-mode phase function.
    Agrees with ζ only at the eigenvalues (i.e. roots of the characteristic
    equation F(ν) = 0).
    '''
    θ = Θ_g(ν, ΔΠ, ν_g)
    return 1/(1 + ΔΠ / Δν * ν**2 * (q * jnp.cos(θ)**2 + jnp.sin(θ)**2/q))

# Setting up the characteristic function F

@jax.jit
def F(ν, ν_p, ν_g, Δν, ΔΠ, q):
    '''
    Characteristic function F such that F(ν) = 0 yields eigenvalues.
    '''
    return jnp.tan(Θ_p(ν, Δν, ν_p)) * jnp.tan(Θ_g(ν, ΔΠ, ν_g)) - q

@jax.jit
def Fp(ν, ν_p, ν_g, Δν, ΔΠ, qp=0):
    '''
    First derivative dF/dν. Required for some numerical methods.
    '''
    return (
        jnp.tan(Θ_g(ν, ΔΠ, ν_g)) / jnp.cos(Θ_p(ν, Δν, ν_p))**2 * jnp.pi / Δν
        + jnp.tan(Θ_p(ν, Δν, ν_p)) / jnp.cos(Θ_g(ν, ΔΠ, ν_g))**2 * jnp.pi / ΔΠ / ν**2
        - qp
    )

@jax.jit
def Fpp(ν, ν_p, ν_g, Δν, ΔΠ, qpp=0):
    '''
    Second derivative d²F / dν². Required for some numerical methods.
    '''
    return (
        2 * F(ν, ν_p, ν_g, Δν, ΔΠ, 0) / jnp.cos(Θ_p(ν, Δν, ν_p))**2 * (jnp.pi / Δν)**2
        + 2 * F(ν, ν_p, ν_g, Δν, ΔΠ, 0) / jnp.cos(Θ_g(ν, ΔΠ, ν_g))**2 * (jnp.pi / ΔΠ / ν**2)**2
        - 2 * jnp.tan(Θ_p(ν, Δν, ν_p)) / jnp.cos(Θ_g(ν, ΔΠ, ν_g))**2 * jnp.pi / ΔΠ / ν**3
        + 2 / jnp.cos(Θ_p(ν, Δν, ν_p))**2 * jnp.pi / Δν / jnp.cos(Θ_g(ν, ΔΠ, ν_g))**2 * jnp.pi / ΔΠ / ν**2
        - qpp
    )


## UTILS

@jax.jit
def α_to_q(α, Δν, ΔΠ, ν, weak=False):
    '''
    Relation between the off-diagonal elements of the coupling matrix
    α (same units as ω²) and the JWKB coupling coefficient q (dimensionless).

    For derivation of this expression see Ong & Gehan 2023.

    We expect units of Δν and ΔΠ such that Δν * ΔΠ is dimensionless.
    '''
    if not weak:
        return (jnp.sqrt((2 * jnp.pi**2)**2 + (α / (4 * ν**2))**2) - 2 * jnp.pi**2) / (Δν * ΔΠ)
    else:
        return (α / (8 * jnp.pi * ν**2))**2 / Δν / ΔΠ

## COUPLING

@jax.jit
def halley_iteration(x, y, yp, ypp, λ=1.):
    '''
    Halley's method (2nd order Householder):
    x_{n+1} = x_n = 2 f f' / (2 f'² - f f''),
    again with damping.
    '''
    return x - λ * 2 * y * yp / (2 * yp * yp - y * ypp)

@partial(jax.jit, static_argnums=(5,))
def couple(ν_p, ν_g, q_p, q_g, λ=.5, maxiter=50):
    '''
    Solve the characteristic equation with Halley's method.
    This converges even faster than Newton's method and is capable
    of handling quite numerically difficult scenarios with not
    very much damping.
    '''
    Δν = jnp.median(jnp.diff(ν_p))
    ΔΠ = -jnp.median(jnp.diff(1/ν_g))

    def _body(i, x0):
        νm_p, νm_g = x0
        νm_p = halley_iteration(νm_p,
                                F(νm_p, ν_p, ν_g, Δν, ΔΠ, q_p),
                                Fp(νm_p, ν_p, ν_g, Δν, ΔΠ),
                                Fpp(νm_p, ν_p, ν_g, Δν, ΔΠ), λ=λ)
        νm_g = halley_iteration(νm_g,
                                F(νm_g, ν_p, ν_g, Δν, ΔΠ, q_g),
                                Fp(νm_g, ν_p, ν_g, Δν, ΔΠ),
                                Fpp(νm_g, ν_p, ν_g, Δν, ΔΠ), λ=λ)
        return νm_p, νm_g

    νm_p, νm_g = jax.lax.fori_loop(0, maxiter, _body, (ν_p, ν_g))

    return νm_p, νm_g