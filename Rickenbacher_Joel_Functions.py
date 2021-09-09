#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:53:40 2021

@author: joel
"""

# load the required functions
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import multivariate_normal
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import numpy as np


############# Functions for the DGP ##################

def dgp(b, mean, cov, u_sd, n):
    """ Creates one draw of a DGP with multivariate normal covariates, a
    randomly assigning treatment value of one or zero and normal noise.
    Inputs:
        - b: vector (1D-array) of true betas (first one for constant)
        - mean: vector (1D-array) of means from multivariate normal
        - cov: covariance matrix (2D-array) from multivariate normal
        - u_sd: standard deviation of the normally distributed noise variable
    Outputs:
        - x: Matrix of the covariates without treatment
      – x_t: Matrix of the covariates with a randomly assigned treatment.
             Which is either one or zero with 40% probability
        - y: Regression vector (outcomes)
    """
    x = multivariate_normal(mean, cov, n)
    x_t = np.c_[np.random.binomial(1, 0.40, size = n), x]
    y = b[0] + x_t @ b[1:] + np.random.normal(0,u_sd,n)
    return(x, x_t, y)


def dgp2(b, mean, cov, u_sd, n):
    """ Creates one draw of a DGP with one normally distributed covariate, 
    a non-random treatment value, that is dependant on the prior defined 
    covariate as well as some normally distributed noise.
    Inputs:
        - b: vector (1D-array) of true betas (first one for constant)
        - mean: Scalar for the mean of a normally distributed covariate
        - cov: 2-d for the variance of a normally distributed covariate
        - u_sd: standard deviation of the normal noise variable
    Outputs:
        - x: Matrix of the covariate without the treatment
      - x_t: Matrix of the covariate included the treatment
        - y: Regression vector (outcomes)
    """
    x = multivariate_normal(mean, cov, n)
    x_t = np.where(x<10, 0, 1)
    y = b[0] + x @ b[1:] + x_t @ b[1:] + np.random.normal(0,u_sd,n)
    return(x, x_t, y)



############# Functions for the estimators ##################

def ols(x,y):
    """OLS coefficients
    Inputs:
        - x: covariates matrix
        - y: Regression vector (outcomes)
    Output:
    - Betas: Regression coefficients (1D-array)
    """
    n = y.shape[0]          # num of obs
    x_t = np.c_[np.ones(n), x]  # add constant
    betas = np.linalg.inv(x_t.T @ x_t) @ x_t.T @ y  # calculate coeff
    return betas


def ipw(exog, t, y, reps):
    """Average treatment estimates according to the IPW.
    Inputs: 
        - exog: Covariates
        - t: 
        - y: 
     - reps:      
   Output:
     - ate: Average treatment effect
- ate_std : Standard deviation of the average treatment effect
    """
    # convert all passed matrices into pandas data frame or Series
    exog = pd.DataFrame(exog)
    t = pd.Series(t)
    y = pd.Series(y)

    pscores = sm.Logit(endog=t, exog=sm.add_constant(exog)).fit(
        disp=0).predict()

    ate = np.mean((t * y) / pscores - ((1 - t) * y) / (1 - pscores))

    ate_boot = []
    for rep in range(reps):
        boot_sample = np.random.choice(exog.index, size=exog.shape[0], replace=True)
        treat_boot = t.loc[boot_sample]
        y_boot = y.loc[boot_sample]

        # append the ate score for per bootstrap run
        ate_boot.append(np.mean((treat_boot * y_boot) / pscores - ((1 - treat_boot) * y_boot) / (1 - pscores)))

    ate_std = np.std(ate_boot)
    # t_val = ate / ate_std
    # df = pd.DataFrame({"ate": ate, "ate_std": ate_std, "t_value": t_val})
    return [ate, ate_std]


def OLS_DML(Y, X, D):
    
    """Approach in order to compare OLS and DML estimators.
    Inspired by: https://github.com/jgitr/opossum/blob/master/double_machine_learning_example/dml.py
    Inputs:
        - X: covariates matrix
        - Y: Regression vector (outcomes)
        - D: Assignment vector
        Output:
            - Betas: Regression coefficients (1D-array)
            """
    # array to store OLS, naive and cross result
    treatment_est = np.zeros(2)

    N = len(Y)
    num_trees = 50
    # Now run the different methods
    #
    # OLS --------------------------------------------------
    OLS = sm.OLS(Y,D)
    results = OLS.fit()
    treatment_est[0] = results.params[0]

    # Naive double machine Learning ------------------------
    naiveDMLg =RandomForestRegressor(num_trees , max_depth=2)
    # Compute ghat
    naiveDMLg.fit(X,Y)
    Ghat = naiveDMLg.predict(X)
    naiveDMLm =RandomForestRegressor(num_trees , max_depth=2)
    naiveDMLm.fit(X,D)
    Mhat = naiveDMLm.predict(X)
    # vhat as residual
    Vhat = D-Mhat
    treatment_est[1] = np.mean(np.dot(Vhat,Y-Ghat))/np.mean(np.dot(Vhat,D))

    return treatment_est

############# Function for the simulations ##################

def simulation(b, mean, cov, u_sd, sz, mc):
    """Monte-Carlo simulation with the DGP from line 27
    Inputs:
        - b: Vector (1D-array) of true betas (first one for constant)
     - mean: Vector (1D-array) of means from multivariate normal
      - cov: covariance matrix (2D-array) from multivariate normal
     - u_sd: standard deviation of the normally distributed noise variable
       - sz: Sample sizes
       - mc: Number of simulations
    Output:
    - Dataframe MSE from OLS and IPW Estimators
    """
    # set up an empty data frame
    df_mse = pd.DataFrame(columns = ["OLS_MSE", "IPW_MSE", "IPW_std_ATE"])
    num_simulations = mc
    n_list = []
    for i in sz:
        # append the sample size
        n_list.append(i)

        # will become a list of lists, where each element holds three betas
        beta_list = []
        ate_list = []
        # estimate the betas
        for j in range(mc):
            # generate data
            x, x_t, y = dgp(b, mean, cov, u_sd, i)
            # estimate betas and store them in "beta_list"
            beta_list.append(ols(x_t, y))
            ate_list.append(ipw(x, np.choose([0] * i, x_t.T), y, 100))

        # calculate the mean per beta coefficient
        aggr_betas = [np.array(beta_list)[:, j].mean() for j in range(len(beta_list[0]))]
        aggr_ate = [np.array(ate_list)[:, j].mean() for j in range(len(ate_list[0]))]

        # put the average squared deviation of the beta-estimation and the
        # originally set values into the data frame
        # PLUS the "ATE", its Standard Deviation and the t-value
        df_mse.loc[len(n_list) - 1] = [((b[1] - aggr_betas[1])**2) / i] + [((b[1] - aggr_ate[0])**2) / i] + aggr_ate[1:3]

    # insert a sample size column into the data frame
    df_mse.insert(loc = 0, column = "n", value = n_list)
    return df_mse



