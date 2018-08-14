# Nynke Niehof, 2018

import numpy as np
import unittest
from sys import path
from os.path import dirname

path.append(dirname(path[0]))
from Experiment.GVS import GVS


class TestMaxVoltage(unittest.TestCase):

    def test_upper_lim(self):
        self.gvs1 = GVS(max_voltage=5.0)
        self.assertAlmostEqual(self.gvs1.max_voltage, 3.0)
        self.gvs1.quit()

    def test_negative_lim(self):
        self.gvs2 = GVS(max_voltage=-40)
        self.assertAlmostEqual(self.gvs2.max_voltage, 3.0)
        self.gvs2.quit()

    def test_change_upper_lim(self):
        self.gvs3 = GVS(max_voltage=2.5)
        self.gvs3.max_voltage = 10
        self.assertAlmostEqual(self.gvs3.max_voltage, 3.0)
        self.gvs3.quit()

    def test_voltage_below_upper_lim(self):
        self.gvs4 = GVS(max_voltage=0.5)
        self.assertAlmostEqual(self.gvs4.max_voltage, 0.5)
        self.gvs4.quit()


class TestNidaqConnection(unittest.TestCase):

    def test_connect_single_channel(self):
        self.gvs1 = GVS()
        physical_channel_name = "cDAQ1Mod1/ao0"
        timing = {"rate": 1e3, "samps_per_chan": 8000}
        connected = self.gvs1.connect(physical_channel_name, **timing)
        self.assertTrue(connected)
        self.gvs1.quit()

    def test_connect_two_channels(self):
        self.gvs2 = GVS()
        physical_channel_name = ["cDAQ1Mod1/ao0", "cDAQ1Mod1/ao1"]
        timing = {"rate": 1e3, "samps_per_chan": 8000}
        connected = self.gvs2.connect(physical_channel_name, **timing)
        self.assertTrue(connected)
        self.gvs2.quit()

    def test_connect_without_timing_args(self):
        self.gvs3 = GVS()
        physical_channel_name = "cDAQ1Mod1/ao0"
        connected = self.gvs3.connect(physical_channel_name)
        self.assertTrue(connected)
        self.gvs3.quit()


def test_signal():
    """
    Generate a signal with an alternating step from 0 V to 1 V and to -1 V.
    Check the generated voltage with an oscilloscope.
    """
    gvs = GVS(max_voltage=3.0)
    timing = {"rate": 1e3, "samps_per_chan": 8000}
    physical_channel_name = "cDAQ1Mod1/ao0"
    connected = gvs.connect(physical_channel_name, **timing)
    if connected:
        samples = np.concatenate((np.zeros(500), np.ones(1000), np.zeros(500)))
        samples = np.concatenate((samples, -samples, samples, -samples))
        gvs.write_to_channel(samples)
    gvs.quit()


def test_dual_channel():
    """
    Generate a signal with an alternating step from 0 V to 1 V and to -1 V
    and send it to two channels.
    Check the generated voltage with an oscilloscope.
    """
    gvs = GVS(max_voltage=3.0)
    timing = {"rate": 1e3, "samps_per_chan": 8000}
    physical_channel_name = ["cDAQ1Mod1/ao0", "cDAQ1Mod1/ao1"]
    connected = gvs.connect(physical_channel_name, **timing)
    if connected:
        samples = np.concatenate((np.zeros(500), np.ones(1000), np.zeros(500)))
        samples1 = np.concatenate((samples, -samples, samples, -samples))
        samples2 = np.concatenate((-samples, samples, -samples, samples))
        two_chan_samples = np.stack((samples1, samples2), axis=0)
        gvs.write_to_channel(two_chan_samples)
    gvs.quit()


if __name__ == "__main__":
    unittest.main()
    test_signal()
    test_dual_channel()
