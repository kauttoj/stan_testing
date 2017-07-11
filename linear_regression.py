# -*- coding: utf-8 -*-

from numpy import *
Nobs = 50
x_true = random.uniform(0,10, size=Nobs)
y_true = random.uniform(-1,1, size=Nobs)
alpha_true = 0.5
beta_x_true = 1.0
beta_y_true = 10.0
eps_true = 0.5
z_true = alpha_true + beta_x_true*x_true + beta_y_true*y_true
z_obs = z_true + random.normal(0, eps_true, size=Nobs)

from matplotlib import pyplot as plt
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(x_true, z_obs, c=y_true, marker='o')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Z')
plt.subplot(1,2,2)
plt.scatter(y_true, z_obs, c=x_true, marker='o')
plt.colorbar()
plt.xlabel('Y')
plt.ylabel('Z')

model = """
data {
   int<lower=4> N; // Number of data points
   real x[N];      // the 1st predictor
   real y[N];      // the 2nd predictor
   real z[N];      // the outcome
}
parameters {
   real alpha;     // intercept
   real betax;     // x-slope
   real betay;     // y-slope
   real<lower=0> eps;       // dispersion
}
model {
   for (i in 1:N)
      z[i] ~ normal(alpha + betax * x[i] + betay * y[i], eps);
}"""

data = {'N':Nobs, 'x':x_true, 'y':y_true, 'z':z_obs}

import pystan
fit = pystan.stan(model_code=model, data=data, iter=10000, chains=4,n_jobs=1)

print(fit)

samples = fit.extract(permuted=True)
alpha = median(samples['alpha'])
beta_x = median(samples['betax'])
beta_y = median(samples['betay'])
eps = median(samples['eps'])

samples = array([samples['alpha'],samples['betax'], samples['betay'], samples['eps']]).T
print(samples.shape)
import corner
tmp = corner.corner(samples[:,:], labels=['alpha','betax','betay','eps'], 
                truths=[alpha_true, beta_x_true, beta_y_true, eps_true])