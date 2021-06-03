class CurvatureLikelihood(Analytical1DLikelihood):
    def __init__(self, x, func, eta, power, noise=None):
        """
        Custom likelihood function using a probability distribution derived
        from power vs arc curvature.

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        eta: array_like 
            arrays of normalized eta values for the observations
        power: array_like
            arrays of secondary spectrum power as a function of normalized
            eta for the observations
        noise: array_like
            initial noise level estimate for each observation
        """

        super(CurvatureLikelihood, self).__init__(x=x, y=np.zeros(np.shape(x)), func=func)
        
        self.norm_eta = eta
        self.power = power
        self.noise = noise
        self.deta = eta[:,1:] - eta[:,:-1] # array of steps in eta

    def log_likelihood(self):
        efac = self.model_parameters['efac']
        equad = self.model_parameters['equad']
        self.sigma = np.sqrt((self.noise * np.exp(efac))**2 + equad**2)
    
        eta_prob = calculate_curvature_peak_probability(self.power, self.sigma)  # probability from power
        integral = np.sum(eta_prob[:,:-1] * self.deta, axis=1)  # integrated power
        integral = integral.reshape((len(integral), 1))
        eta_prob_norm = eta_prob / integral     # normalize power
        
        ymodel = 10 / np.sqrt(-self.residual)     # convert model eta to normalized value
        l = np.zeros(len(self.norm_eta))
        outside = np.argwhere((ymodel > np.max(self.norm_eta, axis=1)) |
                          (ymodel < np.min(self.norm_eta, axis=1))).flatten()
        inside = np.argwhere((ymodel < np.max(self.norm_eta, axis=1)) &
                          (ymodel > np.min(self.norm_eta, axis=1))).flatten()
        l[outside] = np.exp(-100)   # for eta values outside range of data
        inds = np.argmin(np.abs(self.norm_eta[inside] - 
                                ymodel[inside].reshape((len(ymodel[inside]),1))), axis=1)
        l[inside] = eta_prob_norm[inside,inds]  # likelihoods of model eta values
        return np.sum(np.log(l))
