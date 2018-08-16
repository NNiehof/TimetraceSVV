import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class Drift:

    def __init__(self, mu, sigma, f_sampling):
        # parameter of Gaussian distribution to sample
        self.mu = mu
        self.sigma = sigma

        # sampling and filtering characteristics
        self.f_sampling = f_sampling
        self.f_nyquist = self.f_sampling / 2.0
        self.butter_order = 4

    def _gaussian_samples(self, n_samples):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=n_samples)

    def lowpass_filter(self, n_samples, cutoff):
        self.cutoff_freq = cutoff / self.f_nyquist
        self.gauss_noise = self._gaussian_samples(n_samples)
        b, a = signal.butter(self.butter_order, self.cutoff_freq,
                             btype="lowpass")
        self.signal = signal.filtfilt(b, a, self.gauss_noise, method="gust")
        self._scale_to_stdev()
        return self.signal

    def _scale_to_stdev(self):
        sigma_signal = np.std(self.signal)
        sigma_scale = self.sigma / sigma_signal
        self.signal = self.signal * sigma_scale


def stimulus_plot(xvals=None, stim=None, title=""):
    """
    Plot generated stimulus, here for debugging purposes
    """
    plt.figure()
    if xvals is not None:
        plt.plot(xvals, stim)
    else:
        plt.plot(stim)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title(title)


if __name__ == "__main__":
    drift = Drift(mu=0.0, sigma=10.0, f_sampling=50)
    signal = drift.lowpass_filter(n_samples=2000, cutoff=0.2)
    raw_signal = drift.gauss_noise
    stimulus_plot(xvals=None, stim=raw_signal, title="unfiltered")
    butter_order = drift.butter_order
    stimulus_plot(xvals=None, stim=signal,
                  title="filter cutoff = 0.2 Hz,"
                        " {}th order Butterworth".format(butter_order))
    plt.show()
