import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft


def viz(Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()


# %%# Define signal ####################################
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

plt.plot(xo)
plt.show()
plt.plot(x)
plt.show()

# %%# CWT + SSQ CWT ####################################
Twxo, Wxo, *_ = ssq_cwt(xo)
viz(Twxo, Wxo)

Twx, Wx, *_ = ssq_cwt(x)
viz(Twx, Wx)
