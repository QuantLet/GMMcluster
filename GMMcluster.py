import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.mixture import GaussianMixture

import pandas as pd

result=pd.read_csv('cvc.csv')

vol=result['VCRIX']
cp=result['CRIX']
cl=result['cluster']
X=pd.concat([vol, cp], axis=1)

gmm =GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)


#plt.scatter(cp, vol, c=labels, s=40, cmap='viridis')
plt.scatter(cp, vol, c=cl, s=40, cmap='viridis')
plt.xlabel('CRIX')
plt.ylabel('VCRIX')
plt.grid(False)
ax = plt.gca()
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.set_facecolor('W')
#plt.savefig('Clustering3.png', transparent=True)
plt.savefig('Clustering3.png')
plt.show()