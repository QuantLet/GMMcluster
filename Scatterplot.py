



import pandas as pd
import matplotlib.pyplot as plt



# read data
df = pd.read_csv('cvc.csv')


plt.scatter(df.number, df.CRIX,c=df.cluster)# change it to df.date, it doesn't work
plt.xlabel('Time')
plt.ylabel('VCRIX')
plt.show()



plt.scatter(df.number, df.VCRIX, c=df.cluster)
plt.xlabel('Time')
plt.ylabel('CRIX')
plt.show()