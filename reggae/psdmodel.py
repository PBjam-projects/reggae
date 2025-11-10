import jax.numpy as jnp
from .reggae import reggae
from .asymptotic import asymptotic

class PSDModel(reggae, asymptotic):

    def __init__(self, f, n_orders, lw=1/200, nu_0=None, nu_2=None, *args, **kwargs):
        """ Creates an instance of the spectrum model

        Consists of a set of l=0 modes and a set of multiplets for l=1 and l=2 modes.

        The l=1 modes are treated as mixed. The l=2 more are assumed to not be mixed.
        
        Parameters
        ----------
        f: array-like
            Frequencies at which to compute the model spectrum.
        n_orders
            Number of radial p-mode orders to include in the model.
        lw: float
            The linewidth of the p-modes.
        nu_0: array-like
            The frequencies of l=0 modes, with length n_orders.
        nu_2: array-like
            The frequencies of l=0 modes, with length n_orders.
        """

        self.f = f

        self.n_orders = n_orders
        
        self.lw = lw
        
        self.n_g = None
        
        self.nu_0 = nu_0
        
        self.nu_2 = nu_2

    # def init_psdmodel(self, f, numax, nu_0, d02, env_width, alpha_p=0, lw=1/50):
    #     self.f = f

    #     self.numax = numax
    #     self.dnu = np.median(np.diff(nu_0))
    #     self.nu_0 = nu_0
    #     self.d02 = d02

    #     s = np.median(np.sin(2 * np.pi * nu_0 / self.dnu))
    #     c = np.median(np.cos(2 * np.pi * nu_0 / self.dnu))
    #     self.eps0 = np.arctan2(s, c) / 2 / np.pi

    #     self.n_p = nu_0 / self.dnu - self.eps0
    #     self.nmax = numax / self.dnu - self.eps0
    #     self.env_width = env_width

    #     self.alpha_p = alpha_p
    #     self.lw = lw
    #     self.n_g = None

    def get_nu_0(self, theta_asy):
        """ Return the l=0 p-mode frequencies
        
        Parameters
        ----------
        theta_asy: data class
            Data class containing the asymptotic parameters for p-modes. 

        Returns
        -------
        nu0: array-like
            l=0 mode frequencies.
        """

        if self.nu_0 is not None:
            return self.nu_0
        
        return theta_asy.nu_0(self.n_orders)

    def get_nu_2(self, theta_asy):
        """ Get the l=2 mode frequencies

        Returns
        -------
        nu0: array-like
            l=2 mode frequencies
        """

        if self.nu_2 is not None:
            return self.nu_2

        # TODO: return something if self.nu_2 is None

    def get_d02(self, theta_asy):
        """ Compute the l=0,2 separation
        
        Parameters
        ----------
        theta_asy: data class
            Data class of asymtotic p-mode parameters

        Returns
        -------
        d02: float
            Array of local frequency differences between l=0 and l=2 modes
        """

        if self.nu_2 is not None and self.nu_0 is not None: # use fitted mode freqs if available
            return self.get_nu_0(None) - self.get_nu_2(None)
        
        return 10.**(theta_asy.log_d02)

    def getl1(self, theta_asy, theta_reg, **kwargs):
        """Get the mixed l=1 mode frequencies.

        Parameters
        ----------
        theta_asy: data class
            Data class containing the asymptotic p-mode parameters.
        theta_reg: data class
            Data class containing the g-mode and mixing parameters.

        Returns
        -------
        nu_1: array-like
            An array of n_g + n_p mode frequencies.
        """

        numax = 10.**(theta_asy.log_numax)

        dnu = 10.**(theta_asy.log_dnu)
        
        d02 = self.get_d02(theta_asy)
        
        alpha = 10.**(theta_asy.log_alpha)

        nu_0 = self.get_nu_0(theta_asy)
        
        nmax = theta_asy.nmax()
        
        n_p = theta_asy.n_p(self.n_orders)

        return super().getl1(self.n_g, nu_0, numax, dnu, d02, n_p,
                             theta_reg.d01, theta_reg.dPi0, theta_reg.p_L, 
                             theta_reg.p_D, theta_reg.epsilon_g, theta_reg.alpha_g, 
                             **kwargs)

    def _l1model(self, theta_asy, theta_reg, update_n_g=False, amps=None, dnu_p=0, dnu_g=0):
        """Compute l=1 spectrum model.

        Parameters
        ----------
        theta_asy: data class
            Data class containing the asymptotic p-mode parameters.
        theta_reg: data class
            Data class containing the g-mode and mixing parameters.
        update_n_g: bool, opitonal
            Whether or not to update the list of n_g at each sample. Default is False.
        amps: array-like, optional
            Array of mode amplitude. Default is None.
        dnu_p: float, optional
            Small frequency offset for the p-modes. Default is 0.
        dnu_g: float, optional
            Small frequency offset for the g-mode. Default is 0.
        Returns
        -------
        
        """
        nu_0 = self.get_nu_0(theta_asy)

        numax = 10.**(theta_asy.log_numax)

        dnu = 10.**(theta_asy.log_dnu)
        
        env_width = 10.**(theta_asy.log_env_width)
        
        if update_n_g or self.n_g is None:
            self.n_g = self.select_n_g(numax,
                                      [nu_0[0] - 2*dnu, nu_0[-1] + 2*dnu],
                                      [theta_reg.dPi0/1.2, theta_reg.dPi0*1.2],
                                      [0, 1]
                                      )

        nu_1, zeta = self.getl1(theta_asy, theta_reg, dnu_p=dnu_p, dnu_g=dnu_g)
        
        H = self._P_envelope(nu_1, theta_reg.normalisation, numax, env_width)
        
        if amps is not None:
            H = H * amps
        
        lw = jnp.sqrt((zeta*0+dnu*self.lw)**2 + ((1-zeta) * 10.**(theta_asy.log_mode_width))**2)

        return super().l1model(self.f, nu_1, zeta, dnu, amps=H, lw=lw)

    def l1model(self, theta_asy, theta_reg, **kwargs):
        """ Create an l=1 multiplet

        Constructs a triplet of m=-1, 0, and 1, split by a linear combination of
        the core and envelope rotational splitting. The relative power density is 
        modulated the inclination of the stellar rotation axis.

        Parameters
        ----------
        theta_asy: data class
            Data class containing a sample of the asymptotic parameters for the p-modes.
        theta_reg: data class
            Data class containing a sample of l=1 mixing model parameters.
        kwargs : dict
            Additional keyword arguments to be passed to _l1model. 

        Returns
        -------
        l1model : jnp.array
            The spectrum model of a single l=1 multiplet.
        """

        dnu_g = 10.**theta_reg.log_omega_core /  reggae.nu_to_omega

        dnu_p = 10.**theta_reg.log_omega_env /  reggae.nu_to_omega
        
        inc = theta_reg.inclination
        
        return (self._l1model(theta_asy, theta_reg, dnu_g=0, dnu_p=0, **kwargs) * jnp.cos(inc)**2
        
              + self._l1model(theta_asy, theta_reg, dnu_g=-dnu_g, dnu_p=-dnu_p, **kwargs) * jnp.sin(inc)**2 / 2
              
              + self._l1model(theta_asy, theta_reg, dnu_g= dnu_g, dnu_p=dnu_p,**kwargs) * jnp.sin(inc)**2 / 2
               )
