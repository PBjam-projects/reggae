from functools import partial
import numpy as np
from astropy import units as u
from lightkurve.periodogram import SNRPeriodogram

# optimisation
from scipy.optimize import minimize
from yabox import DE
import dynesty
import jax
import jax.numpy as jnp

#pbjam
import pbjam

# self library imports
from .theta import ThetaAsy, ThetaReg
from .psdmodel import PSDModel
from .reggae import reggae

class DipoleStar:
    '''
    Generalisation of pbjam.star to assist in determination of dipole mixed-mode
    parameters.
    '''

    labels=[r'$\Delta\Pi_0$ (relative)', '$L_0$', '$D_0$', r'$\epsilon_g$',
                r'$\log \omega_\mathrm{core}/\mu$Hz', r'$\delta\epsilon_{p1}$',
                r'$\alpha_g$', r'$i$/rad',r'$\log \omega_\mathrm{env}/\mu$Hz',
                r'Normalisation'
               ]

    def __init__(self, s=None, f=None):
        self.s = s
        self.f = f

    @classmethod
    def from_pbjam(klass, star):
        """ Import and prepare results from PBjam

        Takes instances of the either the pbjam.star or pbjam.modeID.modeIDsampler 
        classes and constructs a residual spectrum, by dividing out the l=2,0 (including
        the background) or just the background respectively.

        Parameters
        ----------
        klass: self
            reggae.DipoleStar class self.
        star: object
            Instance of pbjam.star or pbjam.modeID.modeIDsampler classes. 
        """

        pbjam_model = klass.make_pbjam_model(star)

        if hasattr(star, 'modeID') and hasattr(star.modeID, 'sel'):
            sel = star.modeID.sel
            f = star.f[sel]
            s = star.s[sel]
        else:
            f = star.f
            s = star.s

        self = klass(np.array(s / pbjam_model), f)
        self.s_raw = np.array(s)

        if hasattr(star, 'pg'):
            self.pg = star.pg
            self.pg.power = u.Quantity(self.s)
        else:
            # let's ensure that the normalisations are comparable
            self.s_raw /= np.median(self.s_raw)
            self.s /= np.median(self.s)

            self.pg = SNRPeriodogram(self.f * u.uHz, u.Quantity(self.s))


        # theta_asy

        self.nu0, self.nu2, self.theta_asy = self._prepare_theta_asy(star)

        self.norders = len(self.nu0)
        
        self.nmax = 10**(self.theta_asy.log_numax - self.theta_asy.log_dnu) - self.theta_asy.eps
        
        self.l1model = PSDModel(self.f, self.norders)

        self.bounds = self.get_bounds()
        
        self.l1model.n_g = self.select_n_g()

        self.soften = 1 # Set this to oversampling factor if input power spectrum has been oversampled

        self.ID = getattr(star, 'ID', None)

        return self

    def __call__(self, dynamic=False, **kwargs):

        kwargs = {**dict(periodic=[3], reflective=[7]), **kwargs}
        ndim = ThetaReg.dims + 1
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.ln_like, self.ptform, ndim, **kwargs) 
        else:
            sampler = dynesty.NestedSampler(self.ln_like, self.ptform, ndim, **kwargs)

        self.sampler = sampler
        sampler.run_nested()
        return self.summarise()

    def summarise(self, sampler=None):
        if sampler is None:
            sampler = getattr(self, "sampler", None)

        if sampler is not None:
            self.DYresult = sampler.results
            samples = self.DYresult.samples
            weights = np.exp(self.DYresult.logwt - self.DYresult.logz[-1])
            mean, cov = dynesty.utils.mean_and_cov(samples, weights)
            new_samples = dynesty.utils.resample_equal(samples, weights)

            return {
                'mean': mean,
                'cov': cov,
                'new_samples': new_samples,
                'sampler': sampler,
                'weights': weights
            }

    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):
        """ Turns coordinates on the unit cube into θ_reg object

        Parameters
        ----------
        u: list
            Point location in the unit hypercube.

        Returns
        -------
        x: jnp.array
            Point location in model parameter space.
        """

        θ_reg = ThetaReg.prior_transform(u[:ThetaReg.dims], bounds=self.bounds)

        norm = jnp.array([self.bounds[-1][0] + u[-1] * (self.bounds[-1][1] - self.bounds[-1][0])])
        
        return jnp.concatenate((θ_reg, norm))

    @partial(jax.jit, static_argnums=(0,))
    def inv_ptform(self, θ_reg, norm):
        """  Turns θ_reg object into coordinates on the unit cube

        Parameters
        ----------
        θ_reg: ThetaReg object or array-like
            Point location in model parameters space.
        norm: jnp.array
            Normalisation
        
        Returns: jnp.array
            Point location in the unit hypercube.
        """

        u_reg = ThetaReg.inv_prior_transform(θ_reg, bounds=self.bounds)
        
        u_norm = jnp.array([(norm - self.bounds[-1][0]) / (self.bounds[-1][1] - self.bounds[-1][0])])
        
        return jnp.concatenate((u_reg, u_norm))

    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta):
        """ Construct the model for the likelihood evaluation

        Parameters
        ----------
        theta: jnp.array
            Array of model parameters.

        Returns
        -------
        mod: jnp.array
            Spectrum model.
        """

        return self.l1model.l1model(self.theta_asy, ThetaReg(*theta[:9])) * theta[9] + 1

    @partial(jax.jit, static_argnums=(0,))
    def ln_like(self, theta):
        """ Evaluate the model log-likelihood at theta

        The log-likelihood is the joint probability of observing
        the spectrum given the model.

        Parameters
        ----------
        theta: jnp.array
            Array of model parameters

        Returns
        -------
        lnlike: float
            log-likelihood at theta.
        """

        m = self.model(theta)
        
        return -jnp.sum(jnp.log(m) + self.s / m) / self.soften

    def get_bounds(self):
        """ Set bounds for model parameter priors.
        
        Returns
        -------
        bounds: list
            List of bounds.
        """

        bounds = {'deltaPi0'   : (0.875, 0.925), # 0
                  'p_L'        : (1., 3.), # 1
                  'p_D'        : (0., 1.), # 2
                  'epsilon_g'  : (0., 1.), # 3
                  'omega'      : (-1, -0.25), # 4
                  'd01'        : (0.15, 0.25), # 5
                  'alpha_g'    : (-0.0025, 0.0025),
                  'inclination': (0, jnp.pi/2),
                  'phi'        : (10, 200)}
        
        return list(bounds.values())

    def select_n_g(self):
        """ Select the relevant radial g-mode orders

        Based on prior estimates of the relevant range period spacing and 
        eps_g, computes g-modes that significantly overlap with the envelope.

        These will be used to compute the model.

        Returns
        -------
        n_g: jnp.array
            List of radial orders for the g-modes.
        """

        numax = 10**self.theta_asy.log_numax

        dnu = 10**self.theta_asy.log_dnu
        
        env_width = 10**self.theta_asy.log_env_width

        n = self.norders // 2 + 1
        
        width = max((n + 1) * dnu, 3*env_width)
        
        freq_lims = (numax - width, numax + width)

        dPi0_lims = (self.bounds[0][0], self.bounds[0][1])
        
        eps_lims = (self.bounds[3][0], self.bounds[3][1])
        
        return reggae.select_n_g(numax, freq_lims, dPi0_lims, eps_lims)

    @staticmethod
    def _prepare_theta_asy(star):
        """ Get parameters for asymptotic l=2,0+background model.

        Parameters
        ----------
        star: object
            Either a pbjam.star or pbjam.modeID.modeIDsampler class instance.
        
        Returns
        -------
        nu_0: jnp.array
            l=0 mode frequencies.
        nu_2: jnp.array
            l=2 mode frequencies.
        ThetaAsy: object
            ThetaAsy class instance, containing l=2,0 and background model parameters.
        """

        if hasattr(star, 'modeID'):
            if hasattr(star.modeID, 'l20result'):
                s = lambda x: float(star.modeID.l20result['summary'][x][0])
            else: # dev version of PBJam
                s = lambda x: float(star.result['summary'][x][0])
            
            θ = ThetaAsy(
                    log_numax=np.log10(s('numax')),
                    log_dnu=np.log10(s('dnu')),
                    eps=s('eps_p'),
                    log_d02=np.log10(s('d02')),
                    log_alpha=np.log10(s('alpha_p')),
                    log_hmax=np.log10(s('env_height')),
                    log_env_width=np.log10(s('env_width')),
                    log_mode_width=np.log10(s('mode_width'))
                )
            if hasattr(star, 'AsyFreqModel'):
                nu_0 = star.AsyFreqModel.asymptotic_nu_p(s('numax'), s('dnu'), s('eps_p'), s('alpha_p'))[0]
            else:
                nu_0 = star.modeID.l20model.asymptotic_nu_p(s('numax'), s('dnu'), s('eps_p'), s('alpha_p'))[0]
            
            nu_2 = nu_0 - s('d02')
            
            return nu_0, nu_2, θ

        else:
            nu_0 = np.array([row['mean'] for label, row in star.peakbag.summary.iterrows() if 'l0' in label])

            nu_2 = np.array([row['mean'] for label, row in star.peakbag.summary.iterrows() if 'l2' in label])
            
            d02 = np.median(nu_0 - nu_2)
            
            dnu = np.median(np.diff(nu_0))

            mean = lambda x: float(star.asy_fit.summary.loc[x]['mean'])
            
            return nu_0, nu_2, ThetaAsy(
                    log_numax=mean('numax'),
                    log_dnu=np.log10(dnu),
                    eps=mean('eps_p'),
                    log_d02=np.log10(d02),
                    log_alpha=mean('alpha_p'),
                    log_hmax=mean('env_height'),
                    log_env_width=mean('env_width'),
                    log_mode_width=mean('mode_width')
                )

    @staticmethod
    def make_pbjam_model(star, n_samples=50):
        """
        Construct a spectrum model from PBjam output

        Takes either a pbjam.star or pbjam.modeID.modeIDsampler object from a previous
        PBjam run. This is used to construct a residual spectrum.

        In the case of a pbjam.star object being passed, a set of models are drawn
        and averaged.

        Parameters
        ----------
        star: object
            Either a pbjam.star or pbjam.modeID.modeIDsampler object.
        n_samples: int, optional
            Number of samples to use to construct the mean model. Default is 50.

        Returns
        -------
        mod: jnp.array
            Spectrum model.
        """

        if hasattr(star, 'modeID'): #pbjam2 star object
            if hasattr(star.modeID, 'l20model'): #pbjam dev branch
                return star.modeID.l20model.getMedianModel()
            else:
                return star.result['background']

        else: # PBjam1 star object
            peakbag = star.peakbag

            freq = star.pg.frequency.value

            peakbag.ladder_f = np.array(freq)[None, :]
            
            n = peakbag.ladder_s.shape[0]
            
            par_names = ['l0', 'l2', 'width0', 'width2', 'height0', 'height2', 'back']

            acc = np.zeros((n_samples, peakbag.ladder_f.shape[1]))
            
            for i in range(-n_samples, 0):
                z = peakbag.model(*[peakbag.traces[x][i] for x in par_names])
            
                bg = np.min(z, axis=1)
            
                acc[i] = np.sum(z - bg[:, None], axis=0) +1

            return np.mean(acc, axis=0)

    # optimisation tasks

    def simplex(self, theta_reg, norm, **kwargs):
        """ Find maximum log-likelihood using Nelder-Mead downhill simplex minimization.

        Parameters
        ----------
        theta_reg: dataclass
            A ThetaReg dataclass instance
        norm: jnp.array
            An initial point in model parameter space.
        """

        def fun(θ):
            return -self.ln_like(θ)

        self.simplex_results = minimize(fun, np.concatenate([theta_reg.asarray(), [norm]]), **{'method': 'Nelder-Mead', **kwargs})

    def genetic_algorithm(self, solve_kwargs=None, **kwargs):
        """ Find the maximum likelihood using a genetic algorithm.

        Any keyword arguments are passed to the yabox.DE initializatoin.

        solve_kwarg: dict, optional
            Dictionary of argumentts to pass to the yabox.DE.solve method.
        """
        
        if solve_kwargs is None:
            solve_kwargs = {}

        bounds = [[0, 1]] * (ThetaReg.dims + 1)

        def fun(u):
            return -self.ln_like(self.ptform(u))

        self.de = DE(fun, bounds, **kwargs)

        self.de_results = self.de.solve(show_progress=True, **solve_kwargs)