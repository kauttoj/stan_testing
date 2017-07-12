# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import pystan
from scipy.stats import logistic
import random
import corner

x1_mean = [3.2778e-01,5.8283e-01,-1.1451e+00,-3.1103e-01,-1.7550e+00]
x2_mean = [3.3107e+00,2.6345e+00,3.4563e+00,3.1403e+00,3.9905e+00,3.0149e+00]

x1_vars = 1
x2_vars = 0.5

Nobs = 50
        
X1 = np.zeros((len(x1_mean),Nobs))
X2 = np.zeros((len(x2_mean),Nobs))

N1=len(x1_mean)
N2=len(x2_mean)

for i in range(N1):
    X1[i,:] = np.random.normal(loc=x1_mean[i],scale=x1_vars,size=Nobs)

for i in range(N2):
    X2[i,:] = np.random.normal(loc=x2_mean[i],scale=x2_vars,size=Nobs)

model = """
data {
   int<lower=1> N1; // Number of data points
   int<lower=1> N2; // Number of data points
   int<lower=1> Nobs;   
   real X1[N1,Nobs];      // the 1st predictor
   real X2[N2,Nobs];      // the 1st predictor   
}
parameters {
   real gm1;     // x mean
   real gm2;     // y mean
   
   real<lower=0.0001> gv1;
   real<lower=0.0001> gv2;
   
   real m1[N1];     // y mean
   real m2[N2];     // y mean
   
   real<lower=0.0001> v1;     // 
   real<lower=0.0001> v2;     // 
}
model {
   gm1 ~ uniform(-20,20);
   gm2 ~ uniform(-20,20);
   
   gv1 ~ uniform(0,3);
   gv2 ~ uniform(0,3);
   
   v1 ~ uniform(0,3);
   v2 ~ uniform(0,3);
       
   for (i in 1:N1)
       m1[i] ~ normal(gm1,gv1);
       
   for (i in 1:N1) {       
       for (j in 1:Nobs) {
               X1[i,j] ~ normal(m1[i],v1);
       }
   }
   for (i in 1:N2)
       m2[i] ~ normal(gm2,gv2);    
       
   for (i in 1:N2) {
       for (j in 1:Nobs) {
               X2[i,j] ~ normal(m2[i],v2);
       }
   }
} 
generated quantities {
         real d;
         d = gm1-gm2;
}"""

data = {'N1':N1,'N2':N2, 'Nobs':Nobs,'X1':X1,'X2':X2}

import pystan
fit = pystan.stan(model_code=model, data=data, iter=10000, chains=4,n_jobs=1)

print(fit)

samples = fit.extract(permuted=True)
plt.hist(samples['d'])
print('group mean difference p-val = ',min(sum(samples['d']<0)/len(samples['d']),sum(samples['d']>0)/len(samples['d'])))
