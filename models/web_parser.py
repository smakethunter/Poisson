import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
#funkcje pomocnicze
def take_n_split(string,delimiter,pos):
    s=string.split(delimiter)
    return int(s[pos])
def select_minute(data,minute):
    return data.loc[data['Minutes'] == minute]

data=pd.read_csv('weblog.csv')
data=data.head(50)

data['Minutes']=data['Time'].apply(lambda x: take_n_split(x,':',2))
data['Seconds']=data['Time'].apply(lambda x: take_n_split(x,':',3))
data=select_minute(data,38)
nr_events=len(data['Seconds'])

logs_per_second=[]
for i in range (60):
    n_logs=len(data.loc[data['Seconds']==i])
    logs_per_second.append([i,n_logs])

ax=plt.figure()
plt.scatter(np.array(logs_per_second)[:,0],np.array(logs_per_second)[:,1])
plt.plot(np.array(logs_per_second)[:,0],np.array(logs_per_second)[:,1])
plt.show()
lps=np.array(logs_per_second)
prob_lps=[i/len(lps) for i in lps[:,1]]

fig=plt.figure()
plt.scatter(lps[:,0],prob_lps)

plt.show()

# dystrybuanta
distribution=[]
dist2=[0]
distribution.append(logs_per_second[0])
for i in range(1,len(logs_per_second)):

    distribution.append([i,distribution[i-1][1]+logs_per_second[i][1]/nr_events])
    dist2.append(distribution[i-1][1])
ax2=plt.figure()
dist=np.array(distribution)
#plt.plot(dist[:,0],dist[:,1])
plt.scatter(dist[:,0],dist[:,1])
d=np.array(dist2)

plt.plot(dist[:,0],d,'bo',mfc='none')

plt.show()

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

# fit with curve_fit


parameters, cov_matrix = curve_fit(poisson, lps[:,0],prob_lps)

figure2=plt.figure()
x=np.linspace(0,60,1000)

plt.plot(x, poisson(x, parameters), 'r-', lw=2)
plt.scatter(lps[:,0], poisson(lps[:,0], parameters))
plt.show()

