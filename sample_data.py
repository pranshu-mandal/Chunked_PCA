import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import utils as ut
from core import *

n_samples = 30 # INPUT - number of pixels
length = 30000 # INPUT - length if the TOD

t = np.arange(0, length, 1)
baseline = np.sin(t/(length/5)) + np.cos(t/(length/6)) + np.sin(t/(length/15))

sig = []

# Generate a gaussian peak in the signal
m = 250
window = signal.gaussian(m, std = 40)
# plt.plot(window)
# plt.show()

# add the windows at different positions

gaps = int(length/n_samples) # the gaps betweeen each peaks.

for i in range(n_samples):
    add = np.zeros(t.shape)
    add[int(m/n_samples)+(gaps * i):int(m/n_samples)+(gaps * i) + m] += window # m is added in the beginning to give it some padding
    dash = baseline + add + np.random.normal(0, 0.1, t.shape)
    sig.append(dash)
    # plt.plot(dash, alpha=0.5)


# plt.plot(signal[0])
# plt.show()

######################### Testing the PCA methods on the data ##############################

# PCA method
results_pca_comp = ut.work_pca(sig, n_components = 2)
results_pca_comp = np.array(results_pca_comp)

N = 16
flat = np.convolve(results_pca_comp[4][:, 0], np.ones((N,)) / N, mode='same') # averaging the subtracted data

plt.plot(results_pca_comp[4][:, 2], label='data')
plt.plot(results_pca_comp[4][:, 1], label='common mode')
plt.plot(flat, label='subtracted')
plt.legend()
plt.show()

# ChunkedPCA

sig = np.array(sig)
pca = ChunkPCA(n_components=2) # Create instance
pca.flatten(sig) # faltten the data
pca.sigmaclip(sigma=4) # put the data to sigma clip
##### To see if the masking has been done properly, one can use the plot_clipped method to visualize the masked data.
pca.plot_clipped(pix=4) # see how the sigma clipping worked, and set the sigma levels accordingly.
##### Split data and make sure the chunk matrix is compatible with the next step
pca.split_data(num_chunks=30, show_chunk_matrix=True)
final = pca.chunk_pca()


######################## Plotting the final to see the results ###########################################
N = 16
pix = 4
plt.plot(final[pix][:,2])# - np.mean(final[pix][:,0]))
plt.plot(final[pix][:,1])
# plt.plot(ma.masked_array(data[pix][(0):(70000)], mask[pix][(0):(70000)]), alpha=0.6)
plt.show()

plt.plot(np.convolve(final[pix][:,0], np.ones((N,))/N, mode='same'))
plt.show()

# Comparison
# TODO: 3DSCILER AStrovolume