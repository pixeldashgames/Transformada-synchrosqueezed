import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, Wavelet

# Load the image
image = plt.imread('flower.jpg')
# Convert the image to grayscale
image = np.mean(image, axis=2)


wavelet = Wavelet('morlet', N=2048, dtype='float64')


Tx, *_ = ssq_cwt(image, wavelet=wavelet)


Tx_i = issq_cwt(Tx)

# Plot the original image with grayscale colormap
plt.figure()
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.title('Original Image')

plt.show()

# Plot the first layer of the 3D matrix returned by the cwt with grayscale colormap
plt.figure()
plt.imshow(Tx[:, :, 0])
plt.colorbar()
plt.title('Original Image')

plt.show()

# Plot the image after applying the inverse transform with grayscale colormap
plt.figure()
plt.imshow(Tx_i, cmap='gray')
plt.colorbar()
plt.title('Image After Applying Inverse Transform')

plt.show()