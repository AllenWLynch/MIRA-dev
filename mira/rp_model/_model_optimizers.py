import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.preprocessing import StandardScaler
import time
from scipy.special import gammaln, digamma, xlogy

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
        
        #dL_dz  = a*std*np.array(error.T @ X) - z
        
        jac = np.concatenate(
            [ dL_da[0], dL_dg1[0], dL_dg2[0], dL_db[0], np.array([dL_dtheta]), np.zeros(X.shape[1]) ],#np.squeeze(dL_dz),],
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
            )
        )
        
        
    (a, gamma1, gamma2, b, theta), z = res.x[:5], res.x[5:]
    z = z[np.newaxis,:]

    mu = np.exp(-np.where(is_upstream, gamma1, gamma2)*np.abs(distance))
    std = mu/np.sqrt(reg)
    beta = mu + std*z
    
    return res, (a, beta, b, theta)


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



def fit_models(reg = np.inf, *, X, y, exposure, smoothed, global_features,
                distance, is_upstream ):
    
    n_cells, n_peaks = X.shape

    # 1. fit intercept model
    (b_int, theta_int) = fit_intercept_model( 
                                y = y, 
                                exposure = exposure
                            )
    
    start_nb = time.time()
    # 2. fit RP model with initialized intercept, dispersion
    _, (a, beta, b, theta), (gamma_up, gamma_down) = fit_rp_model(reg, 
                                X = X, y = y, exposure=exposure,
                                distance = distance, is_upstream= is_upstream, 
                                init_params = [0.25, 0.069, 0.069, b_int, theta_int * 2] + [0.]*n_peaks
                            )
    
    
    end_nb = time.time()
    # 3. refit OLS coefficients with smoothed counts
    _, (a, b, theta) = refit_loglinear_coefs(
                             y= y, 
                             exposure = exposure, 
                             smoothed = smoothed, 
                             beta = beta,
                             init_params = (a,b),
                             theta = theta,
                            )
    a = max(a, 0.) # if the unconstrained MLE estimate for a is lt 0, set to 0.
    
    end_refit = time.time()
    # 4. Fix RP model as exposure, regress residuals against global features
    fit_lograte = _predict_log_lambda(a, beta, b, 
                                     X = smoothed, 
                                     exposure = exposure,
                                    )
    
    
    _, (beta_global, theta_global), featurize_fn = fit_global_model(
           global_features = global_features,
           y = y,
           exposure = np.exp(fit_lograte),
           theta = theta,
       )
    
    # refit global dispersion
    end_global = time.time()
    
    return {
        'intercept_model' : (b_int, theta_int),
        'fit_model' : (a, beta, b, theta),
        'saturated_model' : (beta_global, theta_global, featurize_fn),
        'upstream_decay' : _decay_to_distance(gamma_up),
        'downstream_decay' : _decay_to_distance(gamma_down),
    } #, (end_nb - start_nb, end_refit - end_nb, end_global - end_refit)
    


def score(*,intercept_model, fit_model, saturated_model,
                      smoothed, y, global_features, exposure,**kw):
    
    # intercept model logp
    b_int, theta_int = intercept_model
    
    intercept_lograte = np.log(exposure) + b_int
    intercept_logp = _model_log_likelihood(y, np.exp(intercept_lograte), theta_int)

    # RP model logp
    params_fit, theta_fit = fit_model[:3], fit_model[3]
    
    fit_lograte = _predict_log_lambda(*params_fit, 
                            X = smoothed, exposure = exposure)
    
    fit_logp = _model_log_likelihood(y, np.exp(fit_lograte), theta_fit)
    
    # "saturated model" logp
    beta_global, theta_global, feature_fn = saturated_model
    
    saturated_lograte = ( (feature_fn(global_features) @ beta_global.T).reshape(-1) + fit_lograte )
    
    saturated_logp = _model_log_likelihood(y, np.exp(saturated_lograte), theta_global)
    
    return (saturated_logp, fit_logp, intercept_logp), (saturated_lograte, fit_lograte)



def generalized_r2(saturated_logp, fit_logp, intercept_logp):
    return 1 -  (saturated_logp.sum() - fit_logp.sum())/(saturated_logp.sum() - intercept_logp.sum())
    