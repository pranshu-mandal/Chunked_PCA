from core import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip


def make_data(idMKIDs_use, indStart=0, indEnd=-1):  # , idColumn=0):
    res = []
    for idMKID in idMKIDs_use:
        raw_data = tbdata.field("mkid{}".format(idMKID))
        raw_data = np.array(raw_data, dtype=float).transpose()
        dat_FS = raw_data[indStart:-indEnd]
        res.append(dat_FS)
    return res

params = {
    "idMKIDs_use": np.arange(1, 102, 1)
}

hdul = fits.open('101_mkids_rNROs_20180601054821.fits')
tbdata = hdul[1].data
cols = hdul[1].columns

data = make_data(params["idMKIDs_use"], 16 * 255, 16 * 160)  # 16 * 250, 16 * 140
data = np.array(data)

def std_deviation(dat, threshhold):
    mkid_exclude = []
    dat = np.std(dat, axis=1)
    # plt.plot(dat)
    # plt.title('MKID TOD standard deviation')
    # plt.xlabel('MKID number')
    # plt.ylabel('standard devation(arbitrary unit)')
    # plt.hlines(threshhold, 0, len(dat), label='threshold')
    # plt.legend()
    # plt.show()
    # print(len(dat))
    for i in range(len(dat)):
        if dat[i] <= threshhold:
            mkid_exclude.append(i + 1)
    return mkid_exclude

mkid_exclude = std_deviation(data, threshhold=0.003)
# print(len(data))
# print(len(mkid_exclude))
# print (mkid_exclude)

bad_mkids = [48, 71, 75, 78, 79, 81, 87, 90, 97]

for i in bad_mkids:
    # print(i)
    mkid_exclude.remove(i)

params["idMKIDs_use"] = mkid_exclude
data = make_data(params["idMKIDs_use"], 16 * 255, 16 * 160)  # 250, 140
# for i in range(len(data)):
#     m = np.mean(data[i])
#     data[i] = data[i] - m
data = np.array(data)

""" Getting Rid of the jump in data manually.
"""
patch = 1000 # patching length of values
jump_at = 17173 # index at which the jump happened
for i in range(len(data)):
    f = data[i]

    mean_left = np.mean(f[jump_at - patch:jump_at])
    mean_right = np.mean(f[jump_at : jump_at + patch])
    diff_mean = mean_left - mean_right

    f[jump_at:-1] = f[jump_at:-1] + diff_mean
    data[i] = f

print("Data-shape:",data.shape)

############################# Now we have our data to play with #########################################

pca = ChunkPCA(n_components=3) # Create instance
pca.flatten(data) # faltten the data
pca.sigmaclip(sigma=4) # put the data to sigma clip
##### To see if the masking has been done properly, one can use the plot_clipped method to visualize the masked data.
pca.plot_clipped(pix=4) # see how the sigma clipping worked, and set the sigma levels accordingly.
##### Split data and make sure the chunk matrix is compatible with the next step
pca.split_data(num_chunks=50, show_chunk_matrix=True)
final = pca.chunk_pca()


######################## Plotting the final to see the results ###########################################
N = 16
pix = 41
plt.plot(final[pix][:,2])# - np.mean(final[pix][:,0]))
plt.plot(final[pix][:,1])
# plt.plot(ma.masked_array(data[pix][(0):(70000)], mask[pix][(0):(70000)]), alpha=0.6)
plt.show()

plt.plot(np.convolve(final[pix][:,0], np.ones((N,))/N, mode='same'))
plt.show()
