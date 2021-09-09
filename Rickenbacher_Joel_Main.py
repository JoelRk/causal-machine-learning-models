#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon April 12 21:49:06 2021

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
import time
from opossum import UserInterface #Please install this package if not available


# set working directory
PATH = (r'/Users/joel/Desktop/MiQEF 3rd Semester/DA2/Indiv. assignment/Selfstudy_code/')
sys.path.append(PATH)
np.set_printoptions(precision=3) # for more compact printing

# load own functions
import Rickenbacher_Joel_Functions as pc


############# DGP's_ Can be skipped ##################
seed(1996) # to ensure replicability

# First with self-made DGP's
'#DGP 1: Experimental setting'
mean = [10, 6, 8, 9]
cov = [[6, 2, 4, 0], [2, 3, 2, 2], [4, 2, 6, 9], [0, 2, 9, 7]]
b = np.array([1, 1, 1, 1, 1, 1])  # true coefficients
u_sd = 0.5
n = 50
x, x_t, y = pc.dgp(b, mean, cov, u_sd, n)

'#DGP 2: CIA Setting'
mean = [10]
cov = [[2]]
b = np.array([1, 1])  # true coefficients
u_sd = 4
n = 50
x2, x_t2, y2 = pc.dgp2(b, mean, cov, u_sd, n)


'#DGP 3: Using the opossum package'

# number of observations N and number of covariates k
N = 500
k = 3
# initilizing class
u = UserInterface(N, k, seed=1996, categorical_covariates = None)
# assign treatment and generate treatment effect inside of class object
u.generate_treatment(random_assignment = False, 
                     assignment_prob = 0.5, # non-random treatment assignment resulting in on average 50% treated observations 
                     constant_pos = True,
                     constant_neg = False,
                     heterogeneous_pos = False, 
                     heterogeneous_neg = False,
                     no_treatment = False, 
                     discrete_heterogeneous = False,
                     treatment_option_weights = None, 
                     intensity = 5)
# generate output variable y and return all 4 variables
y4, x4, assignment, treatment = u.output_data(binary=False, x_y_relation = 'partial_nonlinear_simple')
u.plot_covariates_correlation()
u.get_weights_covariates_to_outputs()



############# Simulations ##################

'#Simulation 1: Comparing OLS - IPW with DGP1'
# First using DGP 1
b = np.array([10, 15, 5, 6, 12, 14]) # true coefficients
mean = [10, 6, 8, 9]
cov = [[6, 2, 4, 0], [2, 3, 2, 2], [4, 2, 6, 9], [0, 2, 9, 7]]
u_sd = 55
sz = range(50, 1050, 50)
mc = 10

sim_1 = pc.simulation(b, mean, cov, u_sd, sz, mc)
sim_1

# plots
plt.plot(sim_1["n"], sim_1["OLS_MSE"])
plt.plot(sim_1["n"], sim_1["IPW_MSE"])
plt.title('Comparison of MSE from OLS (in blue) and from IPW (in orange)')
plt.xlabel('Mean squared error (MSE) ')
plt.ylabel('Sample size from 10 to 1050')
plt.savefig('OLS_IPW_estimators.png')

'#Simulation 2: Comparing OLS - DML: Linear simple relation between y and x'
start_time = time.time()
mc_iterations = 100
# initilizing empty arrays
avg_treatment_effects = np.zeros(mc_iterations)
treatment_estimations = np.zeros((mc_iterations,2))

# iterating to simulate and estimate mc_iterations times
for i in range(mc_iterations):
    if i%10 ==0:
        print(round(i/mc_iterations,2 ))
    # generate data
    u = UserInterface(1000,50)
    u.generate_treatment(random_assignment=False, constant_pos=True, 
                         heterogeneous_pos=False)
    Y, X, assignment, treatment = u.output_data(binary=False, x_y_relation='linear_simple')
    # save true treatment effects
    avg_treatment_effects[i] = np.mean(treatment[assignment==1])
    # save estimations 
    treatment_estimations[i,:] = pc.OLS_DML(Y,X,assignment)
# extract estimations of each method
treatment_ols = treatment_estimations[:,0]
treatment_naive_dml= treatment_estimations[:,1]
 
duration = time.time() - start_time
print('Duration: ' + str(round(duration,2)) + ' sec')

'#Simulation 3: Comparing OLS - DML: Partial nonlinear simple relation between y and x'
# initilizing empty arrays
avg_treatment_effects_partsimpl = np.zeros(mc_iterations)
treatment_estimations = np.zeros((mc_iterations,2))

# iterating to simulate and estimate mc_iterations times
for i in range(mc_iterations):
    if i%10 ==0:
        print(round(i/mc_iterations,2 ))
    # generate data
    u = UserInterface(1000,50)
    u.generate_treatment(random_assignment=False, constant_pos=True, 
                         heterogeneous_pos=False)
    Y, X, assignment, treatment = u.output_data(binary=False, 
                                                x_y_relation='partial_nonlinear_simple')
    # save true treatment effects
    avg_treatment_effects_partsimpl[i] = np.mean(treatment[assignment==1])
    # save estimations 
    treatment_estimations[i,:] = pc.OLS_DML(Y,X,assignment)
# extract estimations of each method
treatment_ols_partsimpl = treatment_estimations[:,0]
treatment_naive_dml_partsimpl = treatment_estimations[:,1]

duration = time.time() - start_time
print('Duration: ' + str(round(duration,2)) + ' sec')

'#Simulation 4: Comparing OLS - DML: Nonlinear simple relation between y and x'
# initilizing empty arrays
avg_treatment_effects_nonlin = np.zeros(mc_iterations)
treatment_estimations = np.zeros((mc_iterations,2))

# iterating to simulate and estimate mc_iterations times
for i in range(mc_iterations):
    if i%10 ==0:
        print(round(i/mc_iterations,2 ))
    # generate data
    u = UserInterface(1000,50)
    u.generate_treatment(random_assignment=False, constant_pos=True, 
                         heterogeneous_pos=False)
    Y, X, assignment, treatment = u.output_data(binary=False, 
                                                x_y_relation='nonlinear_simple')
    # save true treatment effects
    avg_treatment_effects_nonlin[i] = np.mean(treatment[assignment==1])
    # save estimations 
    treatment_estimations[i,:] = pc.OLS_DML(Y,X,assignment)
# extract estimations of each method
treatment_ols_nonlin = treatment_estimations[:,0]
treatment_naive_dml_nonlin = treatment_estimations[:,1]

duration = time.time() - start_time
print('Duration: ' + str(round(duration,2)) + ' sec')



'#Plotting the results of the main simulation'

fig, axes = plt.subplots(1,3,figsize=(12,4)) # create plot

axes[0].set_title('linear simple')
axes[1].set_title('Partial nonlinear simple')
axes[2].set_title('Nonlinear simple')

axes[0].set_xlabel('average treatment effect estimates')
axes[1].set_xlabel('average treatment effect estimates')
axes[2].set_xlabel('average treatment effect estimates')

axes[0].set_ylabel('Density')
axes[1].set_ylabel('Density')
axes[2].set_ylabel('Density')

axes[0].set_xlim((0.5,3))
axes[1].set_xlim((0.5,3))
axes[2].set_xlim((0.5,3))

axes[0].set_ylim((0,4))
axes[1].set_ylim((0,4))
axes[2].set_ylim((0,4))

sns.distplot(treatment_ols, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='OLS')
sns.distplot(treatment_naive_dml, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='DML naive')

sns.distplot(treatment_ols_partsimpl, ax=axes[1], bins = mc_iterations, hist=False, rug=True, label='OLS')
sns.distplot(treatment_naive_dml_partsimpl, ax=axes[1], bins = mc_iterations, hist=False, rug=True, label='DML naive')

sns.distplot(treatment_ols_nonlin, ax=axes[2], bins = mc_iterations, hist=False, rug=True, label='OLS')
sns.distplot(treatment_naive_dml_nonlin, ax=axes[2], bins = mc_iterations, hist=False, rug=True, label='DML naive')


axes[0].axvline(np.mean(avg_treatment_effects), color='r', label='real treat')
axes[1].axvline(np.mean(avg_treatment_effects), color='r', label='real treat')
axes[2].axvline(np.mean(avg_treatment_effects), color='r', label='real treat')


axes[0].legend()
axes[1].legend()
axes[2].legend()

plt.savefig('dml_estimator_distribution.png')

'#Simulation 5: One identifying assumption is violated in the DGP'
# initilizing empty arrays
avg_treatment_effects_violatedass = np.zeros(mc_iterations)
treatment_estimations = np.zeros((mc_iterations,2))

# iterating to simulate and estimate mc_iterations times
for i in range(mc_iterations):
    if i%10 ==0:
        print(round(i/mc_iterations,2 ))
    # generate data
    u = UserInterface(100,10)
    u.generate_treatment(random_assignment=True, constant_pos=False, 
                         heterogeneous_pos=True)
    Y, X, assignment, treatment = u.output_data(binary=False, 
                                                x_y_relation='partial_nonlinear_simple')
    # save true treatment effects
    avg_treatment_effects_violatedass[i] = np.mean(treatment[assignment==1])
    # save estimations 
    treatment_estimations[i,:] = pc.OLS_DML(Y,X,assignment)
# extract estimations of each method
treatment_ols_violatedass = treatment_estimations[:,0]
treatment_naive_dml_violatedass = treatment_estimations[:,1]


fig, axes = plt.subplots(1,2,figsize=(12,4))

axes[0].set_title('Random treatment assignment')

axes[0].set_xlabel('Average treatment effect estimates')

axes[0].set_ylabel('Density')

axes[0].set_xlim((0.5,3))

axes[0].set_ylim((0,4))

sns.distplot(treatment_ols_violatedass, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='OLS')
sns.distplot(treatment_naive_dml_violatedass, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='DML naive')

axes[0].axvline(np.mean(avg_treatment_effects_violatedass), color='r', label='real treat')

axes[0].legend()

plt.savefig('Estimators_with_violated_identifying_assumption.png')





