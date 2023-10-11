import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.preprocessing import StandardScaler
from scipy.special import gammaln, digamma, xlogy
import logging
from os.path import basename
logger = logging.getLogger(basename(__name__))


def _decay_to_distance(gamma):
    return -np.log(1/2)/gamma

def _model_log_likelihood(y, rate, theta):
    
    return xlogy(y, rate) - (y + theta)*np.log(rate + theta)\
            + gammaln(y + theta) - gammaln(theta) + theta*np.log(theta) - gammaln(y + 1)


def _mme_dispersion(y, rate):
    '''
    Use method of moments to estimate dispersion parameter.
    '''
    
    sample_variance = np.square( y - rate ).sum()/(len(y) - 1)
    alpha_hat = ( sample_variance - np.mean(rate) )/(rate @ rate) * len(y)

    theta = 1/alpha_hat
    
    return theta


def _left_pad(x):
        return np.hstack([np.ones( (x.shape[0], 1) ), x])


def _predict_log_lambda(
            a, beta, b,*,
            X, exposure):
    
    exposure = exposure[:,np.newaxis]
    
    log_linear_model = np.array(X @ beta.T)
    log_lambda = np.log(exposure) + a*log_linear_model + b

    return log_lambda.reshape(-1)


def fit_intercept_model(*, y, exposure):
    
    b_hat = np.log( y.sum() ) - np.log( exposure.sum() )  # poisson MLE for intercept
    
    _lambda = np.exp(b_hat)*exposure

    return b_hat, _mme_dispersion(y, _lambda) 



def fit_rp_model(reg,*,
            X, y, exposure,
            distance, is_upstream, 
            prior_beta =  1, prior_alpha = 0,
            max_time = np.inf,
            init_params = None,**kw):
    
    _, n_peaks = X.shape
    y = y[:,np.newaxis]
    distance = distance[np.newaxis,:]
    exposure = exposure[:,np.newaxis]

    def _objective_jac(params):

        (a, gamma1, gamma2, b, theta), z = params[:5], params[5:]
        z = z[np.newaxis,:]

        mu = np.exp(-np.where(is_upstream, gamma1, gamma2)*np.abs(distance))
        std = mu/np.sqrt(reg)
        beta = mu + std*z

        log_linear_model = np.array(X @ beta.T)

        log_lambda = np.log(exposure) + a*log_linear_model + b
        _lambda = np.exp(log_lambda)
        
        # negative binomial likelihood with regularizers
        obj_val = y.T @ log_lambda - (y + theta).T @ np.log(_lambda + theta)\
            - 0.5*np.sum(np.square(z))\
            -prior_beta/gamma1 - (prior_alpha + 1)*np.log(gamma1)\
            -prior_beta/gamma2 - (prior_alpha + 1)*np.log(gamma2)\
            + gammaln(y + theta).sum() - len(y)*gammaln(theta) + len(y)*theta*np.log(theta)\
        
        # jacobian
        error = theta/(_lambda + theta) * (y - _lambda)

        dL_da = error.T @ log_linear_model

        dL_dg1 = a*error.T @ np.array( X @ (-distance*beta*is_upstream).T ) \
                    + prior_beta/np.square(gamma1) - (prior_alpha + 1)/gamma1

        dL_dg2 = a*error.T @ np.array( X @ (-distance*beta*(~is_upstream)).T )\
                    + prior_beta/np.square(gamma2) - (prior_alpha + 1)/gamma2
        
        dL_db = np.sum(error, keepdims=True)
        
        dL_dtheta = digamma(y + theta).sum() - len(y)*digamma(theta) - np.sum( (y + theta)/(_lambda + theta) )\
                    - np.log(_lambda + theta).sum() + len(y)*(1 + np.log(theta))
        
        dL_dz  = a*std*np.array(error.T @ X) - z
        
        jac = np.concatenate(
            [ dL_da[0], dL_dg1[0], dL_dg2[0], dL_db[0], np.array([dL_dtheta]), np.squeeze(dL_dz)],
            axis = 0
        )

        return -obj_val, -jac

    
    if init_params is None:
        init_params = [0.25, 0.069, 0.069, -10., 1.] + [0.]*distance.shape[-1]
    
    res = minimize(
            _objective_jac,
            init_params,
            jac = True,
            method = 'tnc',
            bounds = Bounds(
                [0.,1e-3,1e-3, -np.inf, 1e-3] + [-np.inf]*n_peaks, 
                [np.inf]*(n_peaks+5), 
                keep_feasible=True
            ),
            options = dict(maxiter = 250, maxfun = 500),
        )
    
    if not res.success:
        logger.warning(f"RP model optimization failed: {res.message}")
        
        
    (a, gamma1, gamma2, b, theta), z = res.x[:5], res.x[5:]
    z = z[np.newaxis,:]

    mu = np.exp(-np.where(is_upstream, gamma1, gamma2)*np.abs(distance))
    std = mu/np.sqrt(reg)
    beta = mu + std*z
    
    return res, (a, beta, b, theta), (gamma1, gamma2), z


def _fit_nb_regression(*, y, exposure, features, theta,
                     init_params = None):
    
    y = y[:,np.newaxis]
    exposure = exposure[:,np.newaxis]
    n_features = features.shape[1]

    def _objective_jac(params):

        beta = params[np.newaxis,:]
        
        log_lambda = np.log(exposure) + features @ beta.T
        _lambda = np.exp(log_lambda)
        
        # negative binomial likelihood with regularizers
        obj_val = y.T @ log_lambda - (y + theta).T @ np.log(_lambda + theta)
        
        # jacobian
        error = theta/(_lambda + theta) * (y - _lambda)
        
        dL_dbeta = error.T @ features #- 2*beta
        
        jac = np.squeeze(dL_dbeta)
        
        return -obj_val, -jac
    
        
    def _hess(params):
        
        beta = params[np.newaxis,:]
        
        log_lambda = np.log(exposure) + features @ beta.T
        _lambda = np.exp(log_lambda)
        
        w = -theta * _lambda * (y + theta)/np.square(_lambda + theta)
        
        hess = (w * features).T @ features #- 2
        
        return -hess
    
    
    if init_params is None:
        init_params = [0.]*n_features
    
    res = minimize(
            _objective_jac,
            init_params,
            jac = True,
            method = 'newton-cg',
            hess = _hess,
        )
    
    return res, res.x[np.newaxis,:]



def fit_global_model(*,y, exposure, global_features, theta,
                     init_params = None):
    
    scaler = StandardScaler()
    global_features = _left_pad(scaler.fit_transform(global_features))
    
    res, beta = _fit_nb_regression(y = y, 
                       exposure = exposure, 
                       features = global_features, 
                       theta = theta,
                       init_params = init_params,
                      )
    
    lambda_hat = np.exp( ( global_features @ beta.T ).reshape(-1) + np.log(exposure) )
    
    theta_new = _mme_dispersion(y, lambda_hat)
    
    def featurize_fn(x):
        return _left_pad(scaler.fit_transform(x))
    
    return res, (beta, theta_new), featurize_fn  # results, (global_beta, theta), featurization fn



def refit_loglinear_coefs(*, y, exposure, smoothed, theta, beta, init_params = None):
    
    log_linear_model = np.array(smoothed @ beta.T)
    features = _left_pad(log_linear_model)
    
    res, new_coefs = _fit_nb_regression(y = y, 
                                       exposure = exposure, 
                                       features = features, 
                                       theta = theta,
                                       init_params = init_params
                                      )
    
    lambda_hat = np.exp( ( features @ new_coefs.T ).reshape(-1) + np.log(exposure) )
    
    theta_new = _mme_dispersion(y, lambda_hat)
    
    return res, (res.x[1], res.x[0], theta_new) # results, (a, b, theta)