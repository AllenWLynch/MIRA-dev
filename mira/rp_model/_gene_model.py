import numpy as np
import time
from ._model_optimizers import fit_intercept_model, fit_rp_model, refit_loglinear_coefs, \
    fit_global_model, _predict_log_lambda, _model_log_likelihood, _decay_to_distance

import logging
from os.path import basename
logger = logging.getLogger(basename(__name__))


_cell_features = ['X','y','exposure','smoothed','global_features']

def _split_features(features, mask):
    return {
        k : v[mask].copy() if k in _cell_features else v.copy()
        for k,v in features.items()
        }
    

def fit_models(reg = np.inf, 
               regression_path = True,
               seed = 0, train_proportion = 0.7,*, 
               X, y, exposure, smoothed, global_features,
               distance, is_upstream,
               **kw):
    
    n_cells, n_peaks = X.shape

    # 1. fit intercept model
    (b_int, theta_int) = fit_intercept_model( 
                                y = y, 
                                exposure = exposure
                            )
    
    start_nb = time.time()
    rp_model_kw = dict(
        X = X, y = y, exposure=exposure,
        distance = distance, is_upstream=is_upstream,                   
    )

    init_params = [0.25, 0.069, 0.069, b_int, theta_int * 2] + [0.]*n_peaks
    # 2. fit RP model with initialized intercept, dispersion
    
    if not regression_path:
        _, (a, beta, b, theta), (gamma_up, gamma_down), z = fit_rp_model(reg, init_params=init_params, **rp_model_kw)
    
    else:
        trainset_mask = np.random.RandomState(seed).rand(len(y)) < train_proportion
        train, test = _split_features(rp_model_kw, trainset_mask), _split_features(rp_model_kw, ~trainset_mask)
        
        regs = [np.inf] + [2**p for p in range(10,-11, -2)] + [0.]
        models, logps, Zs = [],[],[]
        for reg in regs:
             
            res, (a, beta, b, theta), (gamma_up, gamma_down), z = fit_rp_model(reg, init_params=init_params, **train)
            
            test_lograte = _predict_log_lambda(a, beta, b, 
                                    X = test['X'], 
                                    exposure = test['exposure'],
                                )
            
            logps.append( sum(_model_log_likelihood(test['y'], np.exp(test_lograte), theta)) )
            models.append( (a,beta,b,theta,gamma_up,gamma_down) )
            Zs.append(z)

            init_params = res.x

            logger.info(f"reg = {reg}, logp = {logps[-1]}")

            if len(logps) >1 and logps[-1] < max(logps[:-1]) +  max(logps[:-1])/200:
                logger.info('Early stopping based on held-out log-probability.')
                break

        logger.info('Logps: [{}]'.format(', '.join([f'{p:.2f}' for p in logps])))
             
        a,beta,b,theta,gamma_up,gamma_down = models[np.nanargmax(logps)]
        z = Zs[np.nanargmax(logps)]
    
    
    end_nb = time.time()
    # 3. refit OLS coefficients with smoothed counts
    _, (a, b, theta) = refit_loglinear_coefs(
                             y= y, 
                             exposure = exposure, 
                             smoothed = smoothed, 
                             beta = beta,
                             init_params = (b,a),
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
        'activation_z' : z,
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



def fit_and_score(reg = np.inf, *,train_mask, **features):
    
    
    model = fit_models(reg = reg, **_split_features(features, train_mask))
    
    logps, logrates = score(**model, **features)

    return model, logps, logrates


def generalized_r2(saturated_logp, fit_logp, intercept_logp):
    return 1 -  (saturated_logp.sum() - fit_logp.sum())/(saturated_logp.sum() - intercept_logp.sum())
    
