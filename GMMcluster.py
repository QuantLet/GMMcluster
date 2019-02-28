import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


data=pd.read_csv('cvc.csv')
vol=data['VCRIX'] #volatility of CRIX
cp=data['CRIX']
cl=data['cluster']
num=data['number']
X=pd.concat([vol, cp], axis=1)



gmm =GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(cp, vol, c=labels, s=40, cmap='viridis')
plt.xlabel('CRIX')
plt.ylabel('VCRIX')
#plt.show()

#for presenting a coherent result
plt.scatter(cp, vol, c=cl, s=40, cmap='viridis')
plt.xlabel('CRIX')
plt.ylabel('VCRIX')
plt.grid(False)
ax = plt.gca()
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.savefig('Clustering3.png', transparent=True)
plt.show()



