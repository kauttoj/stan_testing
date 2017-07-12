# -*- coding: utf-8 -*-
"""
This is a simple PyStan experiment to estimate parameters of the discount model
when given an empirical binary questionairre data of the type:
    " do you take 100$ now (option A) or 140$ after 30 days (option B)? "

After few dozens such questions with varying initial values and delays, 
we can then estimate parameters of a given discount model.

Model consists of estimated discount parameters (beta) 
and a dispersion parameter (eps) that determines the smooth decision boundary around
assumed discounted value (similar role as Gaussian error in linear modeling)

Note 1: for responses 0=take delayed offer, 1=take instant offer
Note 2: generalized model commented out

Created on Tue Jul 11 11:43:31 2017
@author: Jannek
"""
import numpy as np
from matplotlib import pyplot as plt
import pystan
from scipy.stats import logistic
import random
import corner

plt.close('all')

# for artificial data generation
initial_vals=[1] # zero delay sum, can be simply taken as 1
delays=[2,7,11,15,25] # delays to be sampled
N=100 # how many samples/questions
M=5

beta=np.array([0.12,0.05,0.20,0.18,0.40]) # discount coefficient (>0)
eps = np.array([0.10,0.20,0.30,0.40,0.30]) # dispersion scale (0 gives deterministic model)


# Some real data (answers are quite consistent, not too noisy)
REAL_delays = [0.50,1.00,1.00,1.00,2.00,2.00,2.00,2.00,2.00,0.50,0.50,0.50,0.50,0.50,1.00,1.00,1.00,2.00,0.50,0.50,0.50,1.00,2.00,0.50,0.50,1.00,1.00,2.00,2.00,0.50,1.00,1.00,1.00,2.00,2.00,2.00,0.50,0.50,1.00,1.00,1.00,2.00,2.00,2.00,0.50,0.50,0.50,1.00,1.00,2.00,0.50,1.00,2.00,2.00]
REAL_offers = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.10,1.10,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.20,1.20,1.20,1.20,1.20,1.25,1.25,1.25,1.25,1.25,1.25,1.30,1.30,1.30,1.30,1.30,1.30,1.30,1.40,1.40,1.40,1.40,1.40,1.40,1.40,1.40,1.50,1.50,1.50,1.50,1.50,1.50,1.60,1.60,1.60,1.60]
REAL_result = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

USE_REAL_DATA = False # False

# discount model definition
# this function gives the positive delay offer that is equivalent to zero delay offer
# in other words, it's zero-delay offer multiplied by inverse discount coefficient
def model(v,d,beta=beta):
    v=np.array(v)
    d=np.array(d)
    y = v*(1+beta[0]*d); # hyperbolic (inverse of the discount)
    #y = v*(1+beta[0]*d)**(beta[1]/beta[0]); # generalized hyperbolic    
    return y

# get random response based on logit function (to generate data)
def generate_answer(init,delay,choice):
    loc=model(init,delay)
    p=1-logistic.cdf(choice,loc=loc,scale=eps)
    res=np.random.binomial(1,p)
    return (loc,res)

plt.figure(1)

if not USE_REAL_DATA:    

    data=np.zeros(shape=(N,5))
    
    # get some upper limit for offers
    m=1.75*model(1.0,max(delays))
    
    # sample artificial data (you can give real data here)
    k=0
    for k in range(N):
        init=random.choice(initial_vals)
        delay=random.choice(delays)
        choice=init+((m-init)*random.random())
        ans=generate_answer(init,delay,choice)    
        data[k,:]=(init/init,delay,choice/init,ans[0]/init,ans[1])
    
    print('data generated')
    
    # plot real line
    plt.plot(delays,list(map(lambda x: model(1,x), delays)))

else:    
    N=len(REAL_delays)
    data=np.zeros(shape=(N,5))
    data[:,1]=REAL_delays
    data[:,2]=REAL_offers
    data[:,4]=REAL_result
    delays=list(set(REAL_delays))

i=data[:,4]==0
plt.plot(data[i,1],data[i,2],'ro')
i=data[:,4]==1
plt.plot(data[i,1],data[i,2],'bo')

# define model, we assume flat prior for now (you can do better)
stan_model = """
data {
   int<lower=10> N; // Number of data points
   real<lower=0> delay[N];      // delays
   int resp[N];                 // responses
   real<lower=1.0> offer[N];    // offers
}
parameters {
//   vector<lower=0.001,upper=1>[2] beta; // discount params
   vector<lower=0.001,upper=1>[1] beta; // discount params
   real<lower=0.001,upper=5> eps;   // dispersion
}
model {
   //eps ~ uniform(0,20);
   //beta ~ uniform(0,2);
   real pos;
   real theta;   
   for (n in 1:N) {
      //pos = pow(1.0+beta[1]*delay[n],beta[2]/beta[1]);
      pos = 1.0 + beta[1]*delay[n];
      theta = 1-logistic_cdf(offer[n],pos,eps);
      resp[n] ~ binomial(1,theta);
   }
}"""

# data for PyStan
data = {'N':N, 'delay':data[:,1],'offer':data[:,2],'resp':data[:,4].astype(int)}

# n_jobs=1 required in Windows :'(
fit = pystan.stan(model_code=stan_model, data=data, iter=15000, chains=4,n_jobs=1)

print(fit)
fit.plot()

# our estimations (median and mean for comparison)
samples = fit.extract(permuted=True)
eps_est = np.median(samples['eps'])
beta_est = np.median(samples['beta'])
eps_est1 = np.mean(samples['eps'])
beta_est1 = np.mean(samples['beta'])

samples = np.array([samples['beta'],samples['eps']]).T
print(samples.shape)

tmp = corner.corner(samples[:,:], labels=['beta','eps'], 
                truths=[beta_est,eps_est])

# plot also estimated boundary 
plt.figure(1)
plt.plot(delays,list(map(lambda x: model(1,x,beta=np.array([beta_est])), delays)))