# Python 3.6

import numpy as np
from sklearn.decomposition import PCA
# import concurrent.futures
# from astropy.stats import sigma_clip

# TODO: use multiprocessing for the chunk_pca method.

def _make_PCA_vector(data, pca_comp):
    """
    This is an internally used function.
    The eigenvector(pca_comp) stands for how much of the data has been accounted for to get the new components in PCA.
    This function multiplies and adds each of the rows with the corresponding eigenvectors for the projection space.

    Parameters
    ----------
    data: arranged in a way such that, each row represents an independent variable(pixels),
        and the values in the row are different observations.
    pca_comp: eigenvector corresponding to a particular component.

    returns: PCA vector for the particular component.
    """
    ndata = len(data)
    res = []
    for i in range(len(data[0])):
        v_pca = 0
        for j in range(ndata):
            v_pca += data[j][i] * pca_comp[j]
        res.append(v_pca)
    return res

def work_pca(data, n_components=2):
    """
    Calculates the common mode for the given dataset.

    Parameters
    ----------
    data: 2D array(numpy.ndarray), rows - TOD for individual pixels. cols - frequency shift value for a given time.
    n_components: n_components for PCA.

    returns: array of shape (data.shape(0), data.shape(1), 3).
        ((data - baseline), baseline, data)
    """
    for i in range(len(data)):
        m = np.mean(data[i])
        data[i] = data[i] - m
    data = np.array(data)
    ndata = len(data)
    features = np.array(data).transpose()
    pca = PCA(n_components=n_components)
    pca.fit(features)
    pca_vectors = []
    results = []
    for i in range(n_components):
        pca_vectors.append(_make_PCA_vector(data, pca.components_[i]))
    for i in range(ndata):  # loop over KIDs
        theData = data[i]
        res = []
        for j in range(len(data[0])):  # loop over TOD
            v = theData[j]
            pc_in = 0
            for k in range(n_components):  # loop over PCs
                dpc = pca_vectors[k][j] * pca.components_[k][i]
                pc_in += dpc
            res.append([v - pc_in, pc_in, v])
        results.append(res)

    return results


def get_flat(results_pca_comp):
    """
    extracts the flat(raw-baseline) data from work_pca results
    :param results_pca_comp: work_pca results
    :return: first element from each work_pca output which is the flattened data
    """
    flat = []
    for i in range(len(results_pca_comp)):
        flat.append(results_pca_comp[i][:, 0])
    flat = np.array(flat)

    return flat

def running_average(flat, N=16, mode = 'same'):
    """
    Calculates running average of a given array

    Parameters
    ----------
    flat: numpy.array, dataset that needs to be averaged
    N: int, Number of samples to convolve.
    mode: str, any mode supported by numpy.convolve

    returns: running average of the same dimension as data.
    """

    running_avged = []

    if np.ndim(flat) == 1:
        running_avged.append(np.convolve(flat, np.ones((N,)) / N, mode=mode))
    elif np.ndim(flat) == 2:
        for i in range(len(flat)):
            running_avged.append(np.convolve(flat[i], np.ones((N,)) / N, mode=mode))

    return running_avged

def make_chunk_mat(num_chunks, mask):
    """
    Creates the chunks matrix,

    Parameters
    ----------
    num_chunks: int, the number of chunks the TOD is needed to be split into.
    mask: 2d mask obtained after sigmaclipping.

    returns: 2d chunk matrix with shape(num_chunks, num_of_pixels)
    """
    len_chunks = int(mask.shape[1] / num_chunks)  # TODO: make it take fraction as well
    len_remaining = mask.shape[1] % num_chunks  # see if there are left over data
    print("remaining data at the end is {}".format(len_remaining))
    chunk_matrix=[]
    if len_remaining != 0:
        chunk_matrix = np.zeros((len(mask), num_chunks + 1))
        for i in range(len(mask)):
            for j in range(num_chunks):
                dot_val = np.prod(~mask[i][(j * len_chunks):(len_chunks * (j + 1))])
                chunk_matrix[i][j] = (~dot_val + 2)
            chunk_matrix[i][num_chunks] = (~(np.prod(~mask[i][(-len_chunks):-1])) + 2)
    elif len_remaining == 0:
        chunk_matrix = np.zeros((len(mask), num_chunks))
        for i in range(len(mask)):
            for j in range(num_chunks):
                dot_val = np.prod(~mask[i][(j * len_chunks):(len_chunks * (j + 1))])
                chunk_matrix[i][j] = (~dot_val + 2)

    t_chunk = chunk_matrix.T

    return t_chunk

def chunk_pca(data, mask, t_chunk):
    """
    The main algorithm for ChunkedPCA.

    Parameters
    ----------
    data: the original data/rawdata.
    mask: the mask array from the sigmaclipped data
    t_chunk: chunk matrix with shape of (num_chunks, num_of_pixels)
    """
    global baseline, final
    len_chunks = int(data.shape[1] / len(t_chunk))
    for i in range(len(t_chunk)) :  # range(len(t_chunk))
        pooled_data = []
        full = []
        empty = []
        for j in range(len(t_chunk[i])): # LET'S CALL IT THE "POOLING METHOD"
            if t_chunk[i][j] == 0:
                val = data[j][(i*len_chunks):(len_chunks*(i+1))]
                pooled_data.append(val - np.mean(val)) #mean centering the pooled_data
                full.append(j)
            elif t_chunk[i][j] == 1:
                empty.append(j)
        print(np.shape(pooled_data), 'number of pixels used, length of chunk')
        result_masked_pca = work_pca(pooled_data, n_components=3) #pca on the pool
        result_masked_pca = np.array(result_masked_pca)

        ##### TODO: make sure the empty doesn't have too much of the pixels, since then the pca won't work.
        print("unused pixels in chunk {}".format(i + 1))
        print(empty)

        ############# Finding the baseline using the most correlated components from the pooled_data #############

        # correlated = []
        #
        # for q in empty:
        #     val = data[q][(i*len_chunks):(len_chunks*(i+1))]
        #     correlation_matrix = np.corrcoef(np.vstack((pooled_data, val)))
        #     fpr = []
        #     for w, t in enumerate(correlation_matrix[-1]):
        #         if t >0:
        #             fpr.append(w)
        #     correlated.append([q, fpr[:-1]])

        ############# Putting the missing pixels values with proper data, baseline #############
        empty_baseline_pool = []
        for k in empty:
            val = data[k][(i*len_chunks):(len_chunks*(i+1))]
            masked_val = np.ma.masked_array(data[k][(i*len_chunks):(len_chunks*(i+1))], mask[k][(i*len_chunks):(len_chunks*(i+1))])
            val = val - np.mean(masked_val)
            empty_baseline_pool = []
            for p in range(result_masked_pca.shape[0]):
                got = result_masked_pca[p][:, 1]
                empty_baseline_pool.append(got)
            baseline = np.mean(empty_baseline_pool, axis=0)
            dat = val - baseline
            average = np.stack([dat, baseline, val], axis=1)
            result_masked_pca = np.insert(result_masked_pca, k, average, 0)

        # print("Final", np.shape(result_masked_pca))

        ##### JOINING ALL THE CHUNKS ######

        if i == 0:
            final = result_masked_pca
        else:
            final = np.concatenate([final, result_masked_pca], axis = 1)
    print("Result data-shape:", np.shape(final))
    return final

def _chunk_pca(data, mask, t_chunk):
    """
    The main algorithm for ChunkedPCA.

    Parameters
    ----------
    data: the original data/rawdata.
    mask: the mask array from the sigmaclipped data
    t_chunk: chunk matrix with shape of (num_chunks, num_of_pixels)
    """
    global baseline, final
    len_chunks = int(data.shape[1] / len(t_chunk))
    for i in range(len(t_chunk)) :  # range(len(t_chunk))
        pooled_data = []
        full = []
        empty = []
        for j in range(len(t_chunk[i])): # LET'S CALL IT THE "POOLING METHOD"
            if t_chunk[i][j] == 0:
                val = data[j][(i*len_chunks):(len_chunks*(i+1))]
                pooled_data.append(val - np.mean(val)) #mean centering the pooled_data
                full.append(j)
            elif t_chunk[i][j] == 1:
                empty.append(j)
        print(np.shape(pooled_data), 'number of pixels used, length of chunk')
        result_masked_pca = work_pca(pooled_data, n_components=3) #pca on the pool
        result_masked_pca = np.array(result_masked_pca)

        ##### TODO: make sure the empty doesn't have too much of the pixels, since then the pca won't work.
        print("unused pixels in chunk {}".format(i + 1))
        print(empty)

        ############# Finding the baseline using the most correlated components from the pooled_data #############

        correlated = []

        for q in empty: 
            val = data[q][(i*len_chunks):(len_chunks*(i+1))]
            correlation_matrix = np.corrcoef(np.vstack((pooled_data, val)))
            fpr = []
            for w, t in enumerate(correlation_matrix[-1]):
                if t >0:
                    fpr.append(w)
            correlated.append([q, fpr[:-1]])

        ############# Putting the missing pixels values with proper data, baseline #############

        for k in empty: 
            val = data[k][(i*len_chunks):(len_chunks*(i+1))]
            masked_val = np.ma.masked_array(data[k][(i*len_chunks):(len_chunks*(i+1))], mask[k][(i*len_chunks):(len_chunks*(i+1))])
            val = val - np.mean(masked_val)
            for x in correlated:
                if x[0] == k:
                    empty_baseline_pool = []
                    for p in x[1]:
                        if p not in empty:
                            got = result_masked_pca[p][:, 1]
                            empty_baseline_pool.append(got)
                    baseline = np.mean(empty_baseline_pool, axis=0)
            dat = val - baseline
            print(dat.shape,'data', baseline.shape, 'baseline', val.shape, 'val')
            average = np.stack([dat, baseline, val], axis=1)        
            result_masked_pca = np.insert(result_masked_pca, k, average, 0)

        # print("Final", np.shape(result_masked_pca))

        ##### JOINING ALL THE CHUNKS ######

        if i == 0:
            final = result_masked_pca
        else:
            final = np.concatenate([final, result_masked_pca], axis = 1)
    print("Result data-shape:", np.shape(final))
    return final