# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import pystan
from scipy.stats import logistic
import random
import corner

Nobs_x = 200
Nobs_y = 300
        
x_true = 0.01
y_true = 0.15
beta_x_true = 0.2
beta_y_true = 0.3

vals_x = np.random.normal(loc=x_true,scale=beta_x_true,size=Nobs_x)
vals_y = np.random.normal(loc=y_true,scale=beta_y_true,size=Nobs_y)

model = """
data {
   int<lower=1> N_x; // Number of data points
   int<lower=1> N_y; // Number of data points
   
   real x[N_x];      // the 1st predictor
   real y[N_y];      // the 1st predictor   
}
parameters {
   real beta_x;     // x mean
   real beta_y;     // y mean
   real<lower=0.0001> alpha_x;     // x dispersion
   real<lower=0.0001> alpha_y;     // y dispersion
}
model {
   beta_x ~ normal(0,1);
   beta_y ~ normal(0,1);
   alpha_x ~ normal(0,5);
   alpha_y ~ normal(0,5);   
       
   for (i in 1:N_x)
      x[i] ~ normal(beta_x,alpha_x);
   for (i in 1:N_y)
      y[i] ~ normal(beta_y,alpha_y);
} 
generated quantities {
    real di;
    di = beta_x - beta_y;           
}"""

data = {'N_x':Nobs_x,'N_y':Nobs_y, 'x':vals_x, 'y':vals_y}

import pystan
fit = pystan.stan(model_code=model, data=data, iter=10000, chains=2,n_jobs=1)

print(fit)

samples = fit.extract(permuted=True)
di = np.median(samples['di'])