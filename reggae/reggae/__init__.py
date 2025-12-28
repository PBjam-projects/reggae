#!/usr/bin/env python3

import numpy as np
#from peakbogging.coupling import generate_matrices
#from peakbogging.coupling import new_modes
#from numpy.polynomial.polynomial import polyval2d
#import scipy.linalg as sla

from functools import partial
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

from .zeta import couple, α_to_q, ζ_p

UNITS = {
        # absolute scalings inherited from Boeing
        'DPI0': 1.25e-4, # μHz¯¹
        'P_L': 4e-3, # Hz²
        'P_D': 6e-4, # dimensionless
        }

class reggae():
    """
    jax'd numerical routines for generation of dipole mixed modes
    and a PSD model. 
    """

    nu_to_omega = 2 * jnp.pi / 1e6

    def __init__(self, ):
        ...

    @staticmethod
    @jax.jit
    def l1model(nu, nu1s, zeta, dnu, *, lw=None, amps=None):
        """
        Generate a power spectrum model from a set of modes (nu1s).

        The model consists of a sequence of Lorentizan profiles.

        Parmeters
        ---------
        nu: jnp.array
            Frequency bins on which to compute the spectrum model.
        nu1s: jnp.array
            l=1 mode frequencies.
        zeta: jnp.array
            Mixing degree for the modes. 0 is completely p-like, 1 is 
            completely g-like.
        dnu: float
            Large separation for the star.
        lw: jnp.array, optional
            Linewidths for the modes. Default is None. 
        amps: jnp.array, optional
            Mode amplitudes. Default is None.
        """

        Hs = jnp.maximum(0, 1. - zeta)
        if amps is not None:
            Hs = Hs * amps

        modewidth1s = lw if lw is not None else (dnu / 200) * jnp.ones_like(nu1s)
        # widths are the same for all modes if not specified

        lorentzians = jnp.zeros_like(nu)

        for i in range(len(nu1s)):
            lorentzians += reggae._lor(nu, nu1s[i], Hs[i], modewidth1s[i])

        return lorentzians

    @staticmethod
    @jax.jit
    def l1model_rot(nu, nu1s, zeta, dnu, omega, *, inc=0, lw=None, amps=None):
        """
        Generate a power spectrum model from a set of modes (nu1s), including the effects of rotation

        The model consists of a sequence of Lorentizan profiles.

        Parmeters
        ---------
        nu: jnp.array
            Frequency bins on which to compute the spectrum model.
        nu1s: jnp.array
            l=1 mode frequencies.
        zeta: jnp.array
            Mixing degree for the modes. 0 is completely p-like, 1 is 
            completely g-like.
        dnu: float
            Large separation for the star.
        omega: float
            Rotation rate of the star.
        inc: float
            Inclination of the rotation axis with respect to the observers line of sigh.
        lw: jnp.array, optional
            Linewidths for the modes. Default is None. 
        amps: jnp.array, optional
            Mode amplitudes. Default is None.
        """

        return (
            reggae.l1model(nu, nu1s, zeta, dnu, lw=lw, amps=amps) * jnp.cos(inc)**2
            + reggae.l1model(nu, nu1s - zeta * omega / reggae.nu_to_omega, zeta, dnu, lw=lw, amps=amps) * jnp.sin(inc)**2 / 2
            + reggae.l1model(nu, nu1s + zeta * omega / reggae.nu_to_omega, zeta, dnu, lw=lw, amps=amps) * jnp.sin(inc)**2 / 2
        )

    @staticmethod
    @jax.jit
    def _lor(nu, nu0, h, w):
        """ Lorentzian to describe a mode.

        Parameters
        ----------
        nu0 : float
            Frequency of lorentzian (muHz).
        h : float
            Height of the lorentizan (SNR).
        w : float
            Full width of the lorentzian (muHz).

        Returns
        -------
        mode : ndarray
            The SNR as a function frequency for a lorentzian.

        """

        return h / (1.0 + 4.0/w**2*(nu - nu0)**2)

    # @staticmethod
    # @jnp.vectorize
    # def _projection(l, m, i):
    #     if l == 0:
    #         if m == 0:
    #             return 1
    #     if l == 1:
    #         if m == 0:
    #             return jnp.cos(i)**2
    #         if jnp.abs(m) == 1:
    #             return jnp.sin(i)**2 / 2
    #     if l == 2:
    #         if m == 0:
    #             return (3 * jnp.cos(i)**2 - 1) ** 2 / 4
    #         if jnp.abs(m) == 1:
    #             return 3/8 * jnp.sin(2 * i)**2
    #         if jnp.abs(m) == 2:
    #             return 3/8 * jnp.sin(i)**4
    #     if l == 3:
    #         if m == 0:
    #             return (3 * jnp.cos(i) + 5 * jnp.cos(3*i))**2 / 64
    #         if jnp.abs(m) == 1:
    #             return 3/64 * (3 + 5 * jnp.cos(2*i))**2 * jnp.sin(i)**2
    #         if jnp.abs(m) == 2:
    #             return 3/8 * 15 / 8 * (jnp.cos(i) * jnp.sin(i)**2)**2
    #         if jnp.abs(m) == 3:
    #             return 5 / 16 * (jnp.sin(i))**6
    #     return 0

    @staticmethod
    @jax.jit
    def asymptotic_nu_g(n_g, dPi0, max_N2, eps_g, alpha=0, numax=0., l=1):
        """
        Asymptotic relation for the g-mode frequencies
        in terms of a fundamental period offset (defined by the
        maximum Brunt-Vaisala frequency), the asymptotic g-mode period
        spacing, the g-mode phase offset, and an optional curvature term.

        Parameters
        ----------
        n_g: array-like
            Array of radial orders at which the g-mode frequencies are computed
        dPi0: float
            The period spacing of the g-modes.
        max_N2: float
            The maximum of the Brunt-Vaisala frequency, used to define the fundamental
            period offset.
        eps_g: float 
            Phase offset for the g-modes.
        alpha: float, optional
            Curvature factor for the g-mode periods. Default is 0.
        numax: float, optional 
            numax of the envelope. Default is 0
        l=1: int, optional
            Angular degree of the g-modes. Default is 1.

        Returns
        -------
        nu_g: jnp.array
            Array of asymptotic g-mode frequencies.
        """

        nu_to_omega = 2 * jnp.pi / 1e6

        P0 = 1 / (jnp.sqrt(max_N2) / nu_to_omega)

        dPi_l = dPi0 / jnp.sqrt(l * (l + 1)) # TODO change this to dPi1

        P_max = jax.lax.cond(numax != 0,
                             lambda numax : 1/numax,
                             lambda numax : 0.,
                             numax)

        n_max = (P_max - P0) / dPi_l - eps_g

        P = P0 + dPi_l * (n_g + eps_g + alpha * (n_g - n_max)**2)

        return 1/P

    @staticmethod
    @jax.jit
    def wrap_polyval2d(x, y, p):
        """
        Evaluate 2D polynomial, broadcasting shapes of x and y where appropriate.

        Parameters
        ----------
        x: array-like
            x-coordinate for 2D polynomial.
        y: array-like
            y-coordinate for 2D polynomial.
        p: array-like
            Polynomial coefficients.
            
        Returns
        -------
        o: array-like
            Result of evaluating the 2D polynomial.
        """

        xx = (x + 0 * y)

        yy = (y + 0 * x)

        shape = xx.shape

        q = reggae.poly_params_2d(p)
        o = reggae.polyval2d(xx.flatten(), yy.flatten(), q)

        return o.reshape(shape)

    @staticmethod
    @jax.jit
    def poly_params_2d(p):
        """
        Turn a parameter vector into an upper triangular coefficient matrix.

        Parameters
        ----------
        p: array-like
            1D array of polynomial coefficients to be broadcast to upper triangular matrix.

        Returns
        -------
        C: array-like
            2D array consisting of a upper triangular matrix with entries corresponding to p.

        """

        # is len(p) a triangular number?
        # n = int((lib.sqrt(1 + 8 * len(p)) - 1) / 2)
        if len(p) == 1:
            n = 1
        if len(p) == 3:
            n = 2
        if len(p) == 6:
            n = 3

        #assert n == lib.floor(n), "length of parameter vector must be a triangular number"
        #n = int(n)
        C = jnp.zeros((n, n))

        # populate matrix

        for i in range(n):
            # ith triangular number as offset
            n_i = i * (i + 1) // 2

            for j in range(i + 1):

                C = C.at[i - j, j].set(p[n_i + j])

        return C

    @staticmethod
    @jax.jit
    def polyval2d(x, y, c, increasing=True):
        """ Evaluate a 2D polynomial

        Parameters
        ----------
        x: array-like
            1D array
        y: array-like
            1D array
        c: array-like
            2D polynomial parameers
        increasing: bool, optional
            Order of the power of the columns.
        """

        m, n = c.shape
        
        X = jnp.vander(x, m, increasing=increasing)
        
        Y = jnp.vander(y, n, increasing=increasing)
        
        return jnp.sum(X[:, :, None] * Y[:, None,: ]*c, axis=(2, 1))

    @staticmethod
    @jax.jit
    def generate_matrices(n_p, n_g, nu_p, nu_g, p_L, p_D):
        """
        Generate a coupling matrix given a set of p- and g-mode frequencies,
        their radial orders n_p and n_g, and some polynomial coefficients for the
        scaled dimensionless coupling strengths and overlap integrals.

        Parameters:
        -----------
        n_p: array-like 
            Contains π-mode radial orders, of length N_π.
        n_g: array-like
            Contains γ-mode radial orders, of length N_γ.
        ν_p: array-like
            Contains p-mode frequencies, of length N_π.
        ν_g: array-like
            Contains g-mode frequencies, of length N_γ.
        p_L: array-like
            Parameter vector describing 2D polynomial coefficients for coupling strengths.
        p_D: array-like
            parameter vector describing 2D polynomial coefficients for overlap integrals.

        Returns
        -------
        L : array-like
            Coupling matrix of shape N_π+N_γ.
        D : array-like
            Coupling matrix of shape N_π+N_γ.
        """

        assert len(n_p) == len(nu_p) and len(n_g) == len(nu_g)

        N_pi = len(n_p)
        N_gamma = len(n_g)

        L_cross = reggae.wrap_polyval2d(n_p[:, jnp.newaxis], n_g[jnp.newaxis, :], p_L) * (nu_g * reggae.nu_to_omega)**2
        D_cross = reggae.wrap_polyval2d(n_p[:, jnp.newaxis], n_g[jnp.newaxis, :], p_D) * (nu_g[jnp.newaxis, :]) / (nu_p[:, jnp.newaxis])

        L = jnp.hstack( (jnp.vstack((jnp.diag(-(nu_p * reggae.nu_to_omega)**2), L_cross.T)),
                         jnp.vstack((L_cross, jnp.diag( -(nu_g * reggae.nu_to_omega)**2 )))))

        D = jnp.hstack((jnp.vstack((jnp.eye(N_pi), D_cross.T)),
                        jnp.vstack((D_cross, jnp.eye(N_gamma)))))

        return L, D

    @staticmethod
    @jax.jit
    def ΔΠ1(dPi0):
        '''
        Compute the dipole g-mode period spacing in inverse μHz,
        given a dimensionless dPi0 parameter.

        Parameters
        ----------
        dPi0: float
            Dimensionless period spacing

        Returns
        -------
        ΔΠ1: Dipole g-mode period spacing in inverse μHz
        '''
        return UNITS['DPI0'] * dPi0 / jnp.sqrt(2)

    @staticmethod
    @jax.jit
    def q(dnu, numax, dPi0, p_L, p_D):
        '''
        Compute the equivalent asymptotic coupling strength between
        p- and g-modes

        Parameters
        ----------
        dnu: float 
            Large separation for the star
        numax: float 
            numax for the star
        dPi0: float
            Dimensionless period spacing
        p_L: float 
            Dimensionless scaled coupling coefficient
        p_D: float 
            Dimensionless scaled coupling coefficient

        Returns
        -------
        q: Equivalent asymptotic coupling strength (dimensionless)
        '''
        ΔΠ1 = reggae.ΔΠ1(dPi0)
        r_L = UNITS['P_L'] * p_L
        r_D = UNITS['P_D'] * p_D
        ω = numax / 1e6 * 2 * jnp.pi
        α = (r_L + r_D) * ω**2

        return α_to_q(α, dnu, ΔΠ1, numax/1e6)

    @staticmethod
    @partial(jax.jit, static_argnums=(12,))
    def getl1(n_g, nu0_p, numax, dnu, d02, n_p, d01, dPi0, p_L, p_D, 
              epsilon_g, alpha_g, asy=False, *, dnu_p=0, dnu_g=0):
        """Compute the l=1 mode frequencies using the matrix formalism

        Parameters
        ----------
        nu0_p: array-like
            Array of l=0 mode frequencies
        numax: float 
            numax for the star.
        dnu: float 
            Large separation for the star.
        d02: float 
            Small separation for the star.
        n_p: array-like 
            Array of radial orders for the p-modes
        d01: float 
            The l=10 separation.
        dPi0: float 
            Period spacing.
        p_L: float 
            Coupling coefficient.
        p_D: float 
            Coupling coefficient
        epsilon_g: float 
            The asymptotic g-mode phase offset.
        alpha_g: float
            The asymptotic g-mode curvature.
        dnu_p: float, optional
            Small frequency offset for the p-modes. Default is 0.
        dnu_g: float, optional
            Small frequency offset for the g-mode. Default is 0.

        Returns
        -------
        nu1: jnp.array
            l=1 mode frequencies.
        zeta: jnp.array
            Mixing degree of the modes.
        """

        nu1_p = nu0_p + dnu / 2 - d02/3 + d01 * dnu + dnu_p # TODO better way to estimate p-like l1's??

        deltaPi0 = UNITS['DPI0'] * dPi0 # in inverse μHz

        r_L = jnp.array([UNITS['P_L'] * p_L])
        r_D = jnp.array([UNITS['P_D'] * p_D])

        nu_g = reggae.asymptotic_nu_g(n_g, deltaPi0, jnp.inf, epsilon_g, numax=numax, alpha=alpha_g) + dnu_g

        if asy:
            ΔΠ1 = reggae.ΔΠ1(dPi0)
            q = reggae.q(dnu, numax, dPi0, p_L, p_D)

            nu1 = jnp.sort(jnp.concatenate(couple(nu1_p, nu_g, q, q)))
            zeta = ζ_p(nu1, q, ΔΠ1, dnu, nu1_p)

        else:
            L, D = reggae.generate_matrices(n_p, n_g, nu1_p, nu_g, r_L, r_D)
            nu1, zeta, E = reggae.new_modes(L, D, n_p)

        return nu1, zeta

    @staticmethod
    @jax.jit
    def new_modes(A, D, n_p, l=1, ξr=None, ξh=None, M=None):
        r"""
        Given the matrices A and D such that we have eigenvectors

        A cᵢ = -ωᵢ² D cᵢ,

        with ω in Hz, we solve for the frequencies ν (μHz), mode mixing coefficient ζ, and
        (if the vector coefficients ξ_r, ξ_h at a specified radius R are provided) the
        inertiae E of the modes of the coupled system. The inertiae are evaluated assuming unit
        normalisation of the eigenfunctions, so that

        Eᵢ = 4π \int dm |ξᵢ(m)|² / (M \int dΩ |ξᵢ(R)|²) === 4π / (M \int dΩ |ξᵢ(R)|²).

        Note that this differs by 4π from some other definitions; this is for consistency
        with GYRE. Overall constant factors shouldn't really matter in any case.

        Parameters
        ----------
        A: array-like
            Coupling matrix.
        D: array-like
            Coupling matrix.
        N_π: int
            Number of p-modes 
        l: int, optional.
            Angular degree, default is 1.
        ξr: array-like
            Radial component of the eigenfunctions, default is None.
        ξh: array-like
            Horizontal component of the eigenfunctions, default is None.
        M: float
            Stellar mass, default is None.
        """

        Λ, U = reggae.eigh(A, D)

        N_pi = len(n_p)

        u = D.shape[0]
        D0 = jnp.zeros((N_pi, u))
        D1 = jnp.hstack((jnp.zeros((u - N_pi, N_pi)), jnp.eye(u - N_pi)))
        D_gamma = jnp.vstack((D0, D1))

        new_omega2 = -Λ
        ζ = jnp.diag(U.T @ D_gamma @ U)

        E = None
        if ξr is not None and ξh is not None and M is not None:
            Λ2 = l*l + l
            ηr = ξr.T @ U
            ηh = ξh.T @ U
            E = 1 / (ηr**2 + Λ2 * ηh**2) / M

        m = jnp.argsort(new_omega2)

        return jnp.sqrt(new_omega2)[m] / reggae.nu_to_omega, ζ[m], E[m] if E is not None else E

    @staticmethod
    @jax.jit
    def _T(x):
        """Helper function of eigh"""
        return jnp.swapaxes(x, -1, -2)

    @staticmethod
    @jax.jit
    def _H(x):
        """Helper function of eigh"""
        return jnp.conj(reggae._T(x))

    @staticmethod
    @jax.jit
    def symmetrize(x):
        """Helper function of eigh"""
        return (x + reggae._H(x)) / 2

    @staticmethod
    @jax.jit
    def standardize_angle(w, b):
        """Helper function of eigh"""
        if jnp.isrealobj(w):
            return w * jnp.sign(w[0, :])

        else:
            # scipy does this: makes imag(b[0] @ w) = 1
            assert not jnp.isrealobj(b)

            bw = b[0] @ w

            factor = bw / jnp.abs(bw)

            w = w / factor[None, :]

            sign = jnp.sign(w.real[0])

            w = w * sign

            return w

    @staticmethod
    @jax.jit
    def eigh(a, b):
        """
        From https://jackd.github.io/posts/generalized-eig-jvp/

        Compute the solution to the symmetrized generalized eigenvalue problem.

        a_s @ w = b_s @ w @ np.diag(v)

        where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized versions of the
        inputs and H is the Hermitian (conjugate transpose) operator.

        For self-adjoint inputs the solution should be consistent with `scipy.linalg.eigh`
        i.e.

        v, w = eigh(a, b)
        v_sp, w_sp = scipy.linalg.eigh(a, b)
        np.testing.assert_allclose(v, v_sp)
        np.testing.assert_allclose(w, standardize_angle(w_sp))

        Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which will be
        slow because there is no GPU implementation of `eig` and it's just a generally
        inefficient way of doing it. Future implementations should wrap cuda primitives.
        This implementation is provided primarily as a means to test `eigh_jvp_rule`.

        Parameters
        ----------
        a: [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
        b: [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

        Returns
        -------
        v: eigenvalues of the generalized problem in ascending order.
        w: eigenvectors of the generalized problem, normalized such that w.H @ b @ w = I.
        """
        a = reggae.symmetrize(a)
        b = reggae.symmetrize(b)
        b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)

        v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)

        v = v.real

        # In Reggae a and b are always real I think? so we can omit the if statement
        # if jnp.isrealobj(a) and jnp.isrealobj(b):
        w = w.real

        # reorder as ascending in w
        order = jnp.argsort(v)

        v = v.take(order, axis=0)

        w = w.take(order, axis=1)

        # renormalize so v.H @ b @ H == 1
        norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)

        norm = jnp.sqrt(norm2)

        w = w / norm

        w = reggae.standardize_angle(w, b)

        return v, w

    @staticmethod
    def select_n_g(numax, freq_lims, dPi0_lims, eps_lims, max_N2=jnp.inf):
        """ Select relevant g-mode radial orders.

        Based on prior estimates for period spacing and g-mode phase offset, computes
        the g-mode radial orders that are expected to be relevant in the sampling.

        Parameters
        ----------
        numax: float
            numax of the star
        freq_lims: list
            List of lower and upper frequency limits to consider.
        dPi0_lims: list
            List of lower and upper limits of the period spacing to consider.
        eps_lims: list
            List of lower and upper limits of the g-mode phase offset to consider.
        max_N2: float
            The maximum of the Brunt-Vaisala frequency, used to define the fundamental
            period offset.
        """
        init_n_g = np.arange(10000)[::-1] + 1

        min_n_g = init_n_g.max()
        max_n_g = init_n_g.min()

        for dDPi0 in jnp.linspace(*dPi0_lims, 3):

            DPi0 = UNITS['DPI0'] * dDPi0

            for eps in jnp.linspace(*eps_lims, 3):

                nu_g = reggae.asymptotic_nu_g(init_n_g, DPi0, max_N2, eps, numax=numax)
                idx = (freq_lims[0] < nu_g) & (nu_g < freq_lims[1])

                t = jnp.where(idx, init_n_g, 0 * init_n_g + jnp.inf).min()
                min_n_g = jnp.minimum(min_n_g, t)

                t = jnp.where(idx, init_n_g, 0 * init_n_g -1).max()
                max_n_g = jnp.maximum(max_n_g, t)

        return jnp.arange(min_n_g, max_n_g)[::-1]
