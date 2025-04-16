import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import utils as ut
from core import *

sig = np.loadtxt("plotted_sample_data.txt")
sig = np.array(sig)
# for i in range(len(sig)):
# 	plt.plot(sig[i], alpha=0.7)
# plt.show()


############## PCA method
results_pca_comp = ut.work_pca(sig, n_components = 2)
results_pca_comp = np.array(results_pca_comp)

N = 16
flat = np.convolve(results_pca_comp[4][:, 0], np.ones((N,)) / N, mode='same') # averaging the subtracted data

# plt.figure(figsize=(18, 8), dpi= 80, facecolor='w', edgecolor='k')

plt.plot(results_pca_comp[4][:, 2], label='data')
plt.plot(results_pca_comp[4][:, 1], label='common mode')
plt.plot(flat, label='subtracted')
# plt.legend()
plt.show()