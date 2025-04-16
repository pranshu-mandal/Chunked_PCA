# Chunked PCA

A Python implementation of Chunked Principal Component Analysis for baseline removal in astronomical time-ordered data (TOD).

## Overview

Chunked PCA is a technique for removing common-mode noise from time-series data, particularly effective for astronomical observations with multiple detectors. By dividing the time stream into chunks and applying PCA separately to each chunk, this method can isolate and remove common-mode signals while preserving unique detector signals.

## Installation

### Prerequisites

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- Astropy

### Install from source

```bash
git clone https://github.com/pranshu-mandal/Chunked_PCA.git
cd Chunked_PCA
pip install -r requirements.txt  
```

## Usage

### Basic usage

```python
from core import ChunkPCA
import numpy as np

# Prepare your data (pixels × time samples)
data = np.array([...])  # Shape should be (n_pixels, n_timesamples)

# Create a ChunkPCA instance
pca = ChunkPCA(n_components=3)  # Number of PCA components to extract

# Preprocess the data
pca.flatten(data)  # Flatten the TOD for sigma clipping
pca.sigmaclip(sigma=4)  # Mask outliers with 4-sigma threshold

# Visualize the masked data to check clipping
pca.plot_clipped(pix=0)  # Look at the first pixel

# Split data into chunks and perform chunked PCA
pca.split_data(num_chunks=50, show_chunk_matrix=True)
result = pca.chunk_pca()

# The result contains for each pixel: [data-baseline, baseline, original_data]
```

### Example with visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# After running ChunkPCA:
N = 16  # Smoothing kernel size
pix = 0  # Pixel to visualize

# Plot the original data and the extracted common mode
plt.figure(figsize=(10, 6))
plt.plot(result[pix][:,2], label='Original data')
plt.plot(result[pix][:,1], label='Common mode')
plt.legend()
plt.show()

# Plot the cleaned data with smoothing
cleaned = np.convolve(result[pix][:,0], np.ones((N,))/N, mode='same')
plt.figure(figsize=(10, 6))
plt.plot(cleaned, label='Cleaned data')
plt.legend()
plt.show()
```

## Method Details

The Chunked PCA algorithm works as follows:

1. **Flattening**: The raw data is first processed with a simple PCA to make the data suitable for sigma clipping.

2. **Sigma Clipping**: Outliers (such as astronomical signals of interest or glitches) are masked using Astropy's sigma_clip function to prevent them from influencing the baseline estimation.

3. **Chunking**: The full time stream is divided into chunks. For each chunk, a validity map is created showing which pixels have clean (unmasked) data in that chunk.

4. **Per-chunk PCA**: For each chunk:
   - Clean data from valid pixels is pooled
   - PCA is performed on this pooled data to extract common-mode patterns
   - For pixels with masked data, the baseline is estimated using the common-mode patterns from valid pixels
   - The estimated baseline is subtracted from each pixel's data

5. **Result Concatenation**: The processed chunks are concatenated to create the final output.

## Use Cases

This method is particularly effective for:

- Multi-detector astronomical time series (e.g., MKID arrays, bolometer arrays)
- Data with transient signals or glitches that need to be preserved
- Datasets with varying baseline drift between detectors


## License

MIT License

Copyright (c) 2023 [Pranshu Mandal]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

[TODO]

## Contact

email: pranshuphy@gmail.com
