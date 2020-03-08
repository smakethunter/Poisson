# import pandas as pd
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.special import factorial
# #funkcje pomocnicze
# def take_n_split(string,delimiter,pos):
#     s=string.split(delimiter)
#     return int(s[pos])
# def select_minute(data,minute):
#     return data.loc[data['Minutes'] == minute]
#
# data=pd.read_csv('weblog.csv')
# data=data.head(50)
#
# data['Minutes']=data['Time'].apply(lambda x: take_n_split(x,':',2))
# data['Seconds']=data['Time'].apply(lambda x: take_n_split(x,':',3))
# data=select_minute(data,38)
# nr_events=len(data['Seconds'])
#
# logs_per_second=[]
# for i in range (60):
#     n_logs=len(data.loc[data['Seconds']==i])
#     logs_per_second.append([i,n_logs])
#
# ax=plt.figure()
# plt.scatter(np.array(logs_per_second)[:,0],np.array(logs_per_second)[:,1])
# plt.plot(np.array(logs_per_second)[:,0],np.array(logs_per_second)[:,1])
# plt.show()
# lps=np.array(logs_per_second)
# prob_lps=[i/len(lps) for i in lps[:,1]]
#
# fig=plt.figure()
# plt.scatter(lps[:,0],prob_lps)
#
# plt.show()
#
# # dystrybuanta
# distribution=[]
# dist2=[0]
# distribution.append(logs_per_second[0])
# for i in range(1,len(logs_per_second)):
#
#     distribution.append([i,distribution[i-1][1]+logs_per_second[i][1]/nr_events])
#     dist2.append(distribution[i-1][1])
# ax2=plt.figure()
# dist=np.array(distribution)
# #plt.plot(dist[:,0],dist[:,1])
# plt.scatter(dist[:,0],dist[:,1])
# d=np.array(dist2)
#
# plt.plot(dist[:,0],d,'bo',mfc='none')
#
# plt.show()
#
# def poisson(k, lamb):
#     return (lamb**k/factorial(k)) * np.exp(-lamb)
#
# # fit with curve_fit
#
#
# parameters, cov_matrix = curve_fit(poisson, lps[:,0],prob_lps)
#
# figure2=plt.figure()
# x=np.linspace(0,60,1000)
#
# plt.plot(x, poisson(x, parameters), 'r-', lw=2)
# plt.scatter(lps[:,0], poisson(lps[:,0], parameters))
# plt.show()
#
# %% md

## Statystyka i procesy stochastyczne, Ćw. proj., grupa 4



# %%

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from IPython.core.display import display
from scipy.optimize import curve_fit
from scipy.special import factorial
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


# funkcje pomocnicze
def take_n_split(string, delimiter, pos):
    s = string.split(delimiter)
    if s[pos][0] == '0':
        return int(s[pos][1])
    else:
        return int(s[pos])


def select_minute(data, minute):
    return data.loc[data['Minutes'] == minute]


def select_hour(data, hour):
    return data.loc[data['Hours'] == hour]


def select_time_data(data, n_samples, hour, minute):
    new_data = data.head(n_samples)
    new_data['Minutes'] = new_data['Time'].apply(lambda x: take_n_split(x, ':', 2))
    new_data['Seconds'] = new_data['Time'].apply(lambda x: take_n_split(x, ':', 3))
    new_data['Hours'] = new_data['Time'].apply(lambda x: take_n_split(x, ':', 1))
    new_data['in_minute'] = new_data['Minutes'].apply(lambda x: x == minute)
    new_data['in_hour'] = new_data['Hours'].apply(lambda x: x == hour)


    data_out = new_data.loc[(new_data['in_minute'] == True) & (new_data['in_hour'] == True)]
    display(data_out)


    return data_out, len(data_out)


# %%

data = pd.read_csv('/home/smaket/PycharmProjects/Logi i poisson/weblog.csv')

selected_minute_data, nr_events = select_time_data(data, 50, 13, 38)
logs_per_second = []
for i in range(60):
    in_minute = selected_minute_data.apply(lambda x: x['Seconds'] == i, axis=1)
    n_logs = len(in_minute[in_minute == True].index)
    logs_per_second.append([i, n_logs])

ax = plt.figure()
plt.scatter(np.array(logs_per_second)[:, 0], np.array(logs_per_second)[:, 1])
plt.plot(np.array(logs_per_second)[:, 0], np.array(logs_per_second)[:, 1])
plt.title("Ilość zgłoszeń na sekundę w ciągu minuty")
plt.show()

# %%

lps = np.array(logs_per_second)
prob_lps = [i / len(lps) for i in lps[:, 1]]

fig = plt.figure()
plt.scatter(lps[:, 0], prob_lps)
plt.title("Prawdopodobieństwo zgłoszenia w danej sekundzie")
plt.show()

# %%

# dystrybuanta
distribution = []
dist2 = [0]
distribution.append(logs_per_second[0])
for i in range(1, len(logs_per_second)):
    distribution.append([i, distribution[i - 1][1] + logs_per_second[i][1] / nr_events])
    dist2.append(distribution[i - 1][1])
ax2 = plt.figure()
dist = np.array(distribution)
# plt.plot(dist[:,0],dist[:,1])
plt.scatter(dist[:, 0], dist[:, 1])
d = np.array(dist2)

plt.plot(dist[:, 0], d, 'bo', mfc='none')
plt.title("Dystrybuanta empiryczna")
plt.show()


# %%

def poisson(k, lamb):
    return (lamb ** k / factorial(k)) * np.exp(-lamb)


# fit with curve_fit


parameters, cov_matrix = curve_fit(poisson, lps[:, 0], prob_lps)

ax = plt.gca(title='Rozkład poissona- krzywa dopasowana')
x = np.linspace(0, 60, 1000)

ax.plot(x, poisson(x, parameters),'r',label=f"$\lambda = {parameters}$")
ax.scatter(lps[:, 0], poisson(lps[:, 0], parameters),label='punkty wyznaczone empirycznie')

ax.legend()
plt.show()

# %% md



