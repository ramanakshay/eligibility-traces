import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

        
        

df = pd.read_csv('history/2024-05-02_22:24:48_None_particle-v0_Contgae/data.csv')
sns.lineplot(x='iteration', y='avg_ep_rew',data=df, label='Clipped Traces')

df = pd.read_csv('history/2024-05-02_22:42:48_40_particle-v0_Contgae/data.csv')
ax = sns.lineplot(x='iteration', y='avg_ep_rew',data=df, label='Accumulating Traces')


ax.set_title('Clipped Trace')
plt.savefig('graphs/ppo.png')
