import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

        
        

df = pd.read_csv('history/2024-05-02_22:55:56_42_particle-v0_Contgae/data.csv')
sns.lineplot(x='iteration', y='avg_ep_rew',data=df, label='lam = 0')

df = pd.read_csv('history/2024-05-02_22:54:54_42_particle-v0_Contgae/data.csv')
ax = sns.lineplot(x='iteration', y='avg_ep_rew',data=df, label='lam = 0.5')

df = pd.read_csv('history/2024-05-02_22:50:41_13_particle-v0_Contgae/data.csv')
ax = sns.lineplot(x='iteration', y='avg_ep_rew',data=df, label='lam = 0.9')

df = pd.read_csv('history/2024-05-02_22:49:03_13_particle-v0_Contgae/data.csv')
ax = sns.lineplot(x='iteration', y='avg_ep_rew',data=df, label='lam = 1.0')

ax.set_title('Continual GAE - Lambda')
plt.savefig('graphs/lambda.png')
