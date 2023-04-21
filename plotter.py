import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv('data.csv').to_numpy()
# time_step = data[:, 0] # Assumes x-values are in first column
# independnet_win_rate = data[:, 1] # Assumes y-values are in second column
# centralized_win_rate = data[:, 2] # Assumes y-values are in second column
# decentralized_win_rate = data[:, 3] # Assumes y-values are in third column
# decentralized_privacy_preserving_summation_win_rate = data[:,4]
# window_size = 5
# independnet_ma = pd.Series(independnet_win_rate).rolling(window_size).mean().values
# centralized_ma = pd.Series(centralized_win_rate).rolling(window_size).mean().values
# decentralized_ma = pd.Series(decentralized_win_rate).rolling(window_size).mean().values
# decentralized_privacy_preserving_summation_ma = pd.Series(decentralized_privacy_preserving_summation_win_rate).rolling(window_size).mean().values
# plt.plot(time_step, independnet_win_rate, alpha=0.2, color='RoyalBlue')
# plt.plot(time_step, centralized_win_rate, label='Vanilla VDN', alpha=0.2, color='MediumPurple')
# plt.plot(time_step, decentralized_win_rate, alpha=0.2, color='Crimson')
# plt.plot(time_step, decentralized_privacy_preserving_summation_win_rate, alpha=0.2, color='DarkCyan')
# plt.plot(time_step, independnet_ma, label=f'Independent Q-Learning', color='RoyalBlue', linewidth=2)
# plt.plot(time_step, centralized_ma, label=f'Point Moving Average (Vanilla VDN)', color='MediumPurple', linewidth=2)
# plt.plot(time_step, decentralized_ma, label=f'PE-VDN I', color='Crimson', linewidth=2)
# plt.plot(time_step, decentralized_privacy_preserving_summation_ma, label=f'PE-VDN II', color='DarkCyan', linewidth=2)
# # plt.legend(loc = 'lower right', ncol = 2)
# #show the grids
# plt.grid()
# plt.xlabel('Total Environment Steps')
# plt.ylabel('Win Rate')
# # plt.show()

# import tikzplotlib
# tikzplotlib.save("plot.tex")

data2 = pd.read_csv('data2.csv').to_numpy()
data = pd.read_csv('data.csv').to_numpy()
time_step = data2[:, 1] # Assumes x-values are in first column
pe_vdn_anchor = data2[:, 2] # Assumes y-values are in second column
pe_vdn_q = data2[:, 3] # Assumes y-values are in third column
pe_vdn_b = data[:,4]
# iql = data[:,1]

window_size = 5
pe_vdn_anchor_ma = pd.Series(pe_vdn_anchor).rolling(window_size).mean().values
pe_vdn_q_ma = pd.Series(pe_vdn_q).rolling(window_size).mean().values
pe_vdn_b_ma = pd.Series(pe_vdn_b).rolling(window_size).mean().values
# iql_ma = pd.Series(iql).rolling(window_size).mean().values

plt.plot(time_step, pe_vdn_anchor, alpha=0.2, color='RoyalBlue')
plt.plot(time_step, pe_vdn_q, alpha=0.2, color='DarkCyan')
plt.plot(time_step, pe_vdn_b, alpha=0.2, color='Crimson')
# plt.plot(time_step, iql, alpha=0.2, color='DarkGray')
plt.plot(time_step, pe_vdn_anchor_ma, label=f'PE-VDN C (Anchor)', color='RoyalBlue', linewidth=2)
plt.plot(time_step, pe_vdn_q_ma, label=f'PE-VDN C (Running)', color='DarkCyan', linewidth=2)
plt.plot(time_step, pe_vdn_b_ma, label=f'PE-VDN B', color='Crimson', linewidth=2)
# plt.plot(time_step, iql_ma, label=f'IQL', color='DarkGray', linewidth=2)
# plt.legend(loc = 'lower right', ncol = 2)
#show the grids
plt.grid()
plt.xlabel('Total Environment Steps')
plt.ylabel('Win Rate')
# plt.show()

import tikzplotlib
tikzplotlib.save("plot.tex")
