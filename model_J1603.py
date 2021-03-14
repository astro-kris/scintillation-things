"""
J1603 model fitting with bilby for arc curvatures only
"""

import sys
sys.path.insert(0, 
  '/home/kris/Documents/research/scintillation_project/scintools-master/scintools')
import glob
import numpy as np
from scint_utils import read_results, float_array_from_dict, \
    read_par, get_earth_velocity, pars_to_params, \
    get_true_anomaly
from scint_models import effective_velocity_annual, arc_curvature
from scint_utils import scint_velocity
import bilby
from bilby.core import result
import pdb

def model_J1603(npoints, anisotropy=True, show=True, save=False, 
                results_file=None):
    
    average = False

    """
    Read results and set parameter arrays
    """
    
    pars = read_par('J1603-7202.par')
    params = pars_to_params(pars)
    
    # Read parameters from file
    params_list = []
    arc_params = read_results('arc_curvature_data.txt')
    params_list.append((arc_params, 'arc'))
    
    '''
    Read in data
    '''
    params_dict = {}
    for p in params_list:
        params_dict['mjd_' + p[1]] = float_array_from_dict(p[0], 'mjd')
        params_dict['tobs_' + p[1]] = float_array_from_dict(p[0], 'tobs') / 86400  # tobs in days
        params_dict['freq_' + p[1]] = float_array_from_dict(p[0], 'freq')
        # params_dict['rcvrs_' +p[1]] = np.array([rcvr[0] for rcvr in (p[0])['name']])
    
    names_arc = np.array(arc_params['name'])
    
    mjd_arc = params_dict['mjd_arc']
    tobs_arc = params_dict['tobs_arc']
    freq_arc = params_dict['freq_arc']
    # rcvrs_arc = params_dict['rcvrs_arc']

    betaeta = float_array_from_dict(arc_params, 'betaeta')
    betaetaerr = float_array_from_dict(arc_params, 'betaetaerr')

    #eta_sqrt_err = 0.5 * betaetaerr * betaeta ** (-3 / 2)
        
    '''
    Data curation
    '''
    mjd_range = [55400, 56500]
    
    #eta_sqrt_error = eta_sqrt_err
    
    # mjd_range_arc = [int(min(mjd_arc)), int(max(mjd_arc))]
    mjd_range_arc = mjd_range

    arc_indices = np.argwhere((betaeta < 50000) &
                              (mjd_arc > mjd_range_arc[0]) &
                              (mjd_arc < mjd_range_arc[1])
                              )
    names_arc = names_arc[arc_indices].squeeze()

    mjd_arc = mjd_arc[arc_indices].squeeze()

    eta = betaeta[arc_indices].squeeze()
    etaerr = betaetaerr[arc_indices].squeeze()
    #eta_sqrt_error = eta_sqrt_err[arc_indices].squeeze()
    
    arc_tobs = tobs_arc[arc_indices].squeeze()
    
        
    if average:
        # Average simultaneous observations
        ii = 0
        mjd_avg = []
        eta_avg = []
        etaerr_avg = []
        while ii < len(mjd_arc):
            imjd = mjd_arc[ii]
            itobs = arc_tobs[ii]
            ind_simul = np.argwhere(np.abs(mjd_arc - imjd) < 30/86400)  # within 30 seconds
            ii += len(ind_simul)
            mjd_avg.append(np.average(mjd_arc[ind_simul], weights=1/etaerr[ind_simul]**2))
            eta_avg.append(np.average(eta[ind_simul], weights=1/etaerr[ind_simul]**2))
            etaerr_avg.append(np.average(etaerr[ind_simul], weights=1/etaerr[ind_simul]**2))
        mjd_arc = np.array(mjd_avg)
        eta = np.array(eta_avg)
        etaerr = np.array(etaerr_avg)
    
    """
    Model the curvature
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd_arc, pars['RAJ'], pars['DECJ'])
    
    print('Getting true anomaly')
    true_anomaly = get_true_anomaly(mjd_arc, params)
    
    weights = 1 / etaerr
    weights = weights.squeeze()
    true_anomaly = true_anomaly.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    
    def eta_model(xdata, cosi, kom, s, d, psi, vism_psi, vism_ra, vism_dec):
        """
        bilby-compatible function for calling arc curvature model
        """
        
        params_ = dict(params)
        
        params_['COSI'] = cosi
        params_['KOM'] = kom
        params_['s'] = s
        params_['d'] = d
        
        if anisotropy:
            params_['psi'] = psi
            params_['vism_psi'] = vism_psi
        else:
            params_['vism_ra'] = vism_ra
            params_['vism_dec'] = vism_dec
        
        ydata = np.zeros(np.shape(xdata))
        weights = np.ones(np.shape(xdata))
            
        model = -arc_curvature(params_, ydata, weights, 
                               true_anomaly,
                               vearth_ra,
                               vearth_dec)
                
        if results_file is None:
            return np.log(model)
        else:
            return model
    
    xdata = mjd_arc
    
    log_etaerr = etaerr / eta
    likelihood = bilby.likelihood.GaussianLikelihood(xdata, np.log(eta), 
                                                     eta_model, log_etaerr)
    injection_parameters = None
    
    #outdir = ''
    #bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)
    
    priors = dict()
    priors['cosi'] = bilby.core.prior.Uniform(-1, 1, 'cosi')
    priors['kom'] = bilby.core.prior.Uniform(0, 360, 'kom', boundary='periodic')
    priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
    priors['d'] = bilby.core.prior.TruncatedGaussian(3.4, 0.5, 0, 100, 'd')
    if anisotropy:
        priors['psi'] = bilby.core.prior.Uniform(0, 180, 'psi', boundary='periodic')
        priors['vism_psi'] = bilby.core.prior.Gaussian(0, 100, 'vism_psi')
        
        # define so don't have to change eta_model args
        priors['vism_ra'] = bilby.core.prior.DeltaFunction(0, 'vism_ra')
        priors['vism_dec'] = bilby.core.prior.DeltaFunction(0, 'vism_dec')
    else:
        priors['vism_ra'] = bilby.core.prior.Gaussian(0, 100, 'vism_ra')
        priors['vism_dec'] = bilby.core.prior.Gaussian(0, 100, 'vism_dec')
        
        # define so don't have to change eta_model args
        priors['psi'] = bilby.core.prior.DeltaFunction(0, 'psi')
        priors['vism_psi'] = bilby.core.prior.DeltaFunction(0, 'vism_psi')
    
    if results_file is None:
        results = bilby.core.sampler.run_sampler(
            likelihood, priors=priors, sampler='dynesty', label='dynesty',
            npoints=npoints, verbose=False, resume=False, 
            outdir='outdir_J1603'.format(npoints))
    else:        
        results = result.read_in_result(filename=results_file, 
                                        outdir=None, label=None, 
                                        extension='json', gzip=False)
        results.plot_with_data(arc_curvature, xdata, eta, ndraws=0, 
                               xlabel='Orbital phase', ylabel='Arc curvature')
    results.plot_corner()
    print(results)
    
    return results
