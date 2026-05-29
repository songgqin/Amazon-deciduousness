"""Savitzky-Golay smoothing utilities.

"""

# In[] Imports
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d

# In[] Functions
def sgfilter_line(vector_in, m1=4, d1=2, m2=4, d2=6, sudden_ratio=0.25):
    rows, columns = vector_in.shape

    savgol_filter_trend = signal.savgol_coeffs(window_length=2 * m1 + 1, deriv=0, polyorder=d1)
    savgol_filter_fine = signal.savgol_coeffs(window_length=2 * m2 + 1, deriv=0, polyorder=d2)

    vector_in = fill_suddendrop_fast(vector_in, sudden_ratio)

    img_NDVI = convolve1d(vector_in, savgol_filter_trend, axis=0, mode='wrap')

    delta = np.abs(vector_in - img_NDVI)
    delta_max = np.max(delta, axis=0)
    delta_max[delta_max == 0] = 1e-6

    Wi = 1.0 * (vector_in >= img_NDVI) + 1.0 * (vector_in < img_NDVI) * (1 - delta / delta_max)
    F0 = np.sum(Wi * np.abs(img_NDVI - vector_in), axis=0)

    F = np.zeros((3, columns))
    F[-1, :] = F0

    img_NDVI = np.maximum(vector_in, img_NDVI)
    img_NDVI_tmp = img_NDVI

    loop_times = 0
    active_mask = np.ones(columns, dtype=bool)

    while np.any(active_mask) and (loop_times < 30):

        img_NDVI_k = convolve1d(img_NDVI_tmp, savgol_filter_fine, axis=0, mode='wrap')

        Fk = np.sum(Wi * np.abs(img_NDVI_k - vector_in), axis=0)

        img_NDVI_tmp = img_NDVI_k
        img_NDVI = np.maximum(img_NDVI_k, vector_in)

        F[0, :] = F[1, :]
        F[1, :] = F[2, :]
        F[2, :] = Fk

        if loop_times >= 2:
            converged = (F[0, :] > F[1, :]) & (F[2, :] > F[1, :])
            active_mask = active_mask & (~converged)

        loop_times += 1

    return img_NDVI, F, loop_times

def fill_suddendrop(vector_in, sudden_ratio=0.4):
    vector_in_pad = np.pad(vector_in, ((1, 1), (0, 0)), 'wrap')
    kernel_edge1 = np.array([1, -1, 0]).reshape((3, 1))
    kernel_edge2 = np.array([0, -1, 1]).reshape((3, 1))
    kernel_mean = np.array([0.5, 0.0, 0.5]).reshape((3, 1))

    sudden_drop1 = np.abs(signal.convolve(vector_in_pad, kernel_edge1, 'valid'))
    sudden_drop2 = np.abs(signal.convolve(vector_in_pad, kernel_edge2, 'valid'))
    vector_fill = signal.convolve(vector_in_pad, kernel_mean, 'valid')

    max_value = np.abs(sudden_ratio * np.resize(np.max(vector_in, axis=0), np.shape(vector_in)))
    sudden_drop_addr = (sudden_drop1 > max_value) & (sudden_drop2 > max_value)
    vector_in[sudden_drop_addr] = vector_fill[sudden_drop_addr]

    return vector_in

def fill_suddendrop_fast(vector_in, sudden_ratio=0.4):
    kernel_edge1 = np.array([1.0, -1.0, 0.0])
    kernel_edge2 = np.array([0.0, -1.0, 1.0])
    kernel_mean = np.array([0.5, 0.0, 0.5])

    sudden_drop1 = np.abs(convolve1d(vector_in, kernel_edge1, axis=0, mode='wrap'))
    sudden_drop2 = np.abs(convolve1d(vector_in, kernel_edge2, axis=0, mode='wrap'))
    vector_fill = convolve1d(vector_in, kernel_mean, axis=0, mode='wrap')

    max_value = np.abs(sudden_ratio * np.max(vector_in, axis=0))
    sudden_drop_addr = (sudden_drop1 > max_value) & (sudden_drop2 > max_value)

    vector_out = np.copy(vector_in)
    vector_out[sudden_drop_addr] = vector_fill[sudden_drop_addr]

    return vector_out

# In[] Main
if __name__ == '__main__':
    print('SmoothTs')
