import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, imshow


def viz(matrix, title):
    plt.figure()
    plt.imshow(matrix, aspect='auto', cmap = 'gray')
    plt.colorbar()
    plt.title(title)
    plt.show()


def apply_cwt(image_str, wavelet_str):
    image = plt.imread(image_str)
    # Convert the image to grayscale
    image = np.mean(image, axis=2)

    # Initialize an empty list to store the results
    Tx_list = []
    for row in image:
        # Apply ssq_cwt to each row
        Tx_row, *_ = ssq_cwt(row, wavelet=wavelet_str)
        Tx_list.append(Tx_row)

    # Convert the list of results into a 3D numpy array
    Tx = np.stack(Tx_list, axis=1)

    # Initialize an empty list to store the results
    Tx_i_list = []
    for row in Tx:
        # Apply issq_cwt to each row
        Tx_i_row = issq_cwt(row, wavelet=wavelet_str)
        Tx_i_list.append(Tx_i_row)

    # Convert the list of results into a 2D numpy array
    Tx_i = np.stack(Tx_i_list, axis=0)

    imshow(Tx[:, :, 0], title='Synchrosqueezed CWT')
    # Plot the original image
    viz(image, 'Original Image')

    # Plot the image after applying the inverse transform with grayscale colormap
    viz(Tx_i, 'Image After Applying Inverse Transform')



# Wavelets used to analise 2D signals
apply_cwt('flower.jpg', 'morlet')
apply_cwt('airplane.jpg', 'bump')

# Wavelet select to see how to behave a wavelet used for 1D signals
apply_cwt('polar_bear.jpg', 'gmw')
