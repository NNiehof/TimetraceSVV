# Nynke Niehof, 2018

import numpy as np
from Experiment.GVS import GVS
from Experiment.GenStim import GenStim
import matplotlib.pyplot as plt


def habituation_signal():
    """
    Generate a habituation signal with a slow ramp
    """
    amp = 2.0
    duration = 25.0
    f_samp = 1e3
    frequency = 1.0
    ramp_length = 10.0
    buffer_size = int(duration * f_samp)
    gvs = GVS(max_voltage=amp)
    timing = {"rate": f_samp, "samps_per_chan": buffer_size}
    connected = gvs.connect("cDAQ1Mod1/ao0", **timing)

    # step stimulus
    make_stim = GenStim(f_samp=f_samp)
    make_stim.step(duration, amp)
    make_stim.fade(f_samp * ramp_length)
    gvs_wave = make_stim.stim

    # make_stim = GenStim(f_samp=f_samp)
    # make_stim.sine(duration, amp, frequency)
    # gvs_wave = make_stim.stim

    if connected:
        print("start galvanic stim")
        gvs.write_to_channel(gvs_wave)
        print("end galvanic stim")
        gvs.quit()

    return gvs_wave


def stimulus_plot(stim, title=""):
    plt.figure()
    plt.plot(stim)
    plt.xlabel("sample")
    plt.ylabel("amplitude (mA)")
    plt.title(title)


if __name__ == "__main__":
    gvs_stim = habituation_signal()
    stimulus_plot(gvs_stim)
    plt.show()

