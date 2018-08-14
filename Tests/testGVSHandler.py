# Nynke Niehof, 2018

import numpy as np
import multiprocessing
import unittest
import sys
from os.path import dirname

sys.path.append(dirname(sys.path[0]))
from Experiment.GVSHandler import GVSHandler


class TestHandlerCommunication(unittest.TestCase):

    def setUp(self):
        self.param_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        queue_manager = multiprocessing.Manager()
        self.log_queue = queue_manager.Queue()
        self.gvsProcess = multiprocessing.Process(target=GVSHandler,
                                                  args=(self.param_queue,
                                                        self.status_queue,
                                                        self.log_queue,
                                                        2001))
        self.status = dict()
        self.gvsProcess.start()
        self.connected = self.status_queue.get()

    def tearDown(self):
        self.param_queue.put("STOP")
        self.gvsProcess.join()
        self.param_queue.close()
        self.status_queue.close()

    def test_connection(self):
        self.assertTrue(self.connected)

    def test_create_stim(self):
        stim = np.ones(2000)
        stim[-1] = 0
        self.param_queue.put(stim)
        while "stim_created" not in self.status:
            self.status = self.status_queue.get()
        self.assertTrue(self.status["stim_created"])

    def test_wrong_param_type(self):
        self.param_queue.put(5)
        while "stim_created" not in self.status:
            self.status = self.status_queue.get()
        self.assertFalse(self.status["stim_created"])

    def test_send_stim(self):
        stim = np.ones(2000)
        stim[-1] = 0
        self.param_queue.put(stim)
        self.param_queue.put(True)
        count = 0
        while count < 10:
            count += 1
            self.status = self.status_queue.get()
            if "stim_sent" in self.status:
                self.assertTrue(self.status["stim_sent"])
                break

    def test_send_duplicate_stim(self):
        stim = np.ones(2000)
        stim[-1] = 0
        self.param_queue.put(stim)
        self.param_queue.put(True)
        count = 0
        while count < 10:
            count += 1
            self.status = self.status_queue.get()
            if "stim_sent" in self.status:
                break
        # attempt to send stimulus a second time
        self.param_queue.put(True)
        count_dup = 0
        while count_dup < 10:
            count_dup += 1
            self.status = self.status_queue.get()
            if "stim_sent" in self.status:
                self.assertFalse(self.status["stim_sent"])
                break


if __name__ == "__main__":
    unittest.main()
