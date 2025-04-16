import numpy as np
import utils as ut
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt

# TODO: flatten function using **median filter**.

class ChunkPCA:
    """
    Removes baseline from a TOD using chunks and PCA.
    """

    def __init__(self, n_components=3):
        self.n_components = n_components

    def flatten(self, data):
        """
        Flatten the TOD so that sigma clipping can be applied
        :param data: TOD 2d dataset
        :return: flat data.
        """
        # First PCA
        self._data = data
        results_pca_comp = ut.work_pca(data, self.n_components)
        results_pca_comp = np.array(results_pca_comp)

        flat = ut.get_flat(results_pca_comp) # get the first component from work_pca
        self._flat = flat
        print("Data flattened")

    def sigmaclip(self, sigma=4, convolve_kernel=16):
        """
        to do the sigma clipping, it is ideal to flatten the TOD such that sigmaclipping can work.
        :param sigma: the sigma threshold level(default = 4).
        :param convolve_kernel: The N in running average, i.e. how many values to use for np.convolve.
        :return: sigma clipped data
        """

        self._sigma = sigma

        flat = ut.running_average(self._flat, N=convolve_kernel)
        self._mask = self._sigmaclip(flat, sigma)
        print("Data sigmaclipped at {}sigma, after convolved by {} samples".format(sigma, convolve_kernel))
        print("To see the sigma clip, use 'plot_clipped' function")
        print("To get the mask, use 'get_mask' function")

    def _sigmaclip(self, data, sigma):
        """
        Perform the digma clipping using astropy sigma_clip
        :param data: flattened data to be clipped
        :param sigma: the sigma threshold level.
        :return: mask
        """
        filtered_data = sigma_clip(data, sigma=sigma, maxiters=5, axis=1)

        mask = np.ma.getmask(filtered_data)
        return mask

    def get_mask(self):
        """
        Call to get the mask.
        :return: mask array
        """
        return self._mask

    def plot_clipped(self, pix=1):
        """
        To visualize how the sigma clipping mask has been applied
        :param pix: which pixel to see
        :return: plot of the
        """

        data = self._data
        mask = self._mask

        plt.plot(np.ma.masked_array(data[pix], ~mask[pix]), 'o', label='Masked data')
        plt.plot(np.ma.masked_array(data[pix], mask[pix]), label='Unmasked data', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Sigma clipping at {}-sigma'.format(self._sigma))
        plt.show()

    def split_data(self, num_chunks, show_chunk_matrix = False):
        """
        Used to split the data into chunks.
        :param num_chunks: how many chunks are to be made
        :return:
        """
        self._t_chunk = ut.make_chunk_mat(num_chunks=num_chunks, mask=self._mask)
        if show_chunk_matrix == True:
            plt.imshow(self._t_chunk.T)
            plt.title("Chunk matrix")
            plt.xlabel("Chunks")
            plt.ylabel("Elements")
            # plt.savefig('chunk_matrix.eps')
            plt.show()

    def chunk_pca(self):
        """
        The main algorithm for chunk PCA
        :return: the array(data-baseline, baseline, data)
        """
        return ut.chunk_pca(self._data, self._mask, self._t_chunk)