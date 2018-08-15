import os
import logging
import multiprocessing
from queue import Empty
import json
import time
import keyboard
import numpy as np
from random import random
from collections import OrderedDict
from psychopy import visual, core, event
from Experiment.GVSHandler import GVSHandler
from Experiment.loggingConfig import Listener, Worker
from Experiment.RandStim import RandStim
from Experiment.GenStim import GenStim

"""
Present constant (step) GVS with a visual line presented at a random start
orientation. Participants can set the orientation of the visual line
to match their subjective visual vertical, so that they see the line as upright.
"""


class StepExp:

    def __init__(self, sj=None, condition=""):

        self.debug = True

        # experiment settings and conditions
        self.sj = sj
        self.paradigm = "stepGVS"
        self.condition = condition
        self.f_sampling = 1e3
        self.screen_refresh_freq = 60
        self.duration_s = 2 # 15.25
        self.visual_only_duration_s = 2# 14.75
        self.visual_soa = None
        self.current_mA = 1.0
        self.physical_channel_name = "cDAQ1Mod1/ao0"
        self.line_ori_step_size = 0.25
        self.line_drift_step_size = 0.1
        self.oled_delay = 0.05
        self.header = "trial_nr; current; line_offset; frame_ori; gvs_start;" \
                      " gvs_end; line_drift; line_ori; frame_time\n"
        self.ramp_duration_s = 0.25

        # longer practice trials
        if "practice" in self.condition:
            self.duration_s = 17.0

        # initialise
        self.make_stim = None
        self.stimuli = None
        self.conditions = None
        self.trials = None
        self.triggers = None
        self.gvs_profile = None
        self.gvs_sent = None
        self.gvs_start = None
        self.gvs_end = None
        self.visual_duration = self.duration_s - (2 * self.ramp_duration_s)
        self.visual_time = np.arange(0, self.visual_duration,
                                     1.0 / self.screen_refresh_freq)
        self.line_orientation = 0.0
        self.frame_ori = None
        self.line_angle = None
        self.visual_onset_delay = self.ramp_duration_s
        self.trial_nr = 0

        # root directory
        abs_path = os.path.abspath("__file__")
        self.root_dir = os.path.dirname(os.path.dirname(abs_path))
        self.settings_dir = "{}/Settings".format(self.root_dir)

    def setup(self):
        # display and window settings
        self._display_setup()

        # set up logging folder, file, and processes
        make_log = SaveData(self.sj, self.paradigm, self.condition,
                            file_type="log", sj_leading_zeros=3,
                            root_dir=self.root_dir)
        log_name = make_log.datafile
        self._logger_setup(log_name)
        main_worker = Worker(self.log_queue, self.log_formatter,
                             self.default_logging_level, "main")
        self.logger_main = main_worker.logger
        self.logger_main.debug("logger set up")

        # set up connection with galvanic stimulator
        self._gvs_setup()
        self._check_gvs_status("connected")

        # trial list
        if "practice" in self.condition:
            conditions_file = "{}/practice_conditions.json".format(
                self.settings_dir)
        else:
            conditions_file = "{}/conditions.json".format(self.settings_dir)
        with open(conditions_file) as json_file:
            self.conditions = json.load(json_file)
        self.trials = RandStim(**self.conditions)
        self.n_trials = self.trials.get_n_trials()

        # create stimuli
        self.make_stim = Stimuli(self.win, self.settings_dir, self.n_trials)
        self.stimuli, self.triggers = self.make_stim.create()
        self.gvs_create = GenStim(f_samp=self.f_sampling)

        # data save file
        self.save_data = SaveData(self.sj, self.paradigm, self.condition,
                                  sj_leading_zeros=3, root_dir=self.root_dir)
        self.save_data.write_header(self.header)

        self.logger_main.info("setup complete")

    def _display_setup(self):
        """
        Window and display settings
        """
        display_file = "{}/display.json".format(self.settings_dir)
        with open(display_file) as json_file:
            win_settings = json.load(json_file)
        self.win = visual.Window(**win_settings)
        self.mouse = event.Mouse(visible=False, win=self.win)

    def _logger_setup(self, log_file):
        """
        Establish a connection for parallel processes to log to a single file.

        :param log_file: str
        """
        # settings
        self.log_formatter = logging.Formatter("%(asctime)s %(processName)s %(thread)d %(message)s")
        if self.debug:
            self.default_logging_level = logging.DEBUG
        else:
            self.default_logging_level = logging.INFO

        # set up listener thread for central logging from all processes
        queue_manager = multiprocessing.Manager()
        self.log_queue = queue_manager.Queue()
        self.log_listener = Listener(self.log_queue, self.log_formatter,
                                     self.default_logging_level, log_file)
        # note: for debugging, comment out the next line. Starting the listener
        # will cause pipe breakage in case of a bug elsewhere in the code,
        # and the console will be flooded with error messages from the
        # listener.
        self.log_listener.start()

    def _gvs_setup(self):
        """
        Establish connection with galvanic stimulator
        """
        buffer_size = int(self.duration_s * self.f_sampling) + 1
        self.param_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        self.gvs_process = multiprocessing.Process(target=GVSHandler,
                                                   args=(self.param_queue,
                                                         self.status_queue,
                                                         self.log_queue,
                                                         buffer_size))
        self.gvs_process.start()

    def _check_gvs_status(self, key, from_queue=None, blocking=True):
        """
        Check the status of *key* on the status queue. Returns a boolean
        for the status. Note: this is a blocking process.
        :param key: str
        :param blocking: bool, set to True to hang until the key parameter
        is found in the queue. Set to False to check the queue once, then
        return.
        :return: bool or None
        """
        if from_queue is None:
            from_queue = self.status_queue
        while True:
            try:
                status = from_queue.get(block=blocking)
                if key in status:
                    return status[key]
            except Empty:
                return None
            if not blocking:
                return None

    def random_walk(self):
        """
        Random walk drift of visual line orientation.
        :return line_drift:
        """
        if random() < 0.5:
            line_drift = -self.line_drift_step_size
        else:
            line_drift = self.line_drift_step_size
        self.line_orientation += line_drift
        return line_drift

    def check_response(self):
        """
        Check for key presses, update the visual line amplitude
        """
        if keyboard.is_pressed("left arrow"):
            self.line_orientation -= self.line_ori_step_size
        elif keyboard.is_pressed("right arrow"):
            self.line_orientation += self.line_ori_step_size
        elif keyboard.is_pressed("esc"):
            self.quit_exp()

    def display_stimuli(self):
        """
        Draw stimuli on screen
        """
        for stim in self.stimuli:
            if self.triggers[stim]:
                self.stimuli[stim].draw()
        self.win.flip()

    def show_visual(self):
        """
        Visual loop that draws the stimuli on screen
        """
        self.triggers["rodStim"] = True
        if self.frame_ori is not None:
            self.triggers["squareFrame"] = True
            self.stimuli["squareFrame"].setOri(self.frame_ori)
        line_start = time.time()

        for frame in self.visual_time:
            # random drift of line
            drift = self.random_walk()
            self.line_drift.append(drift)

            self.stimuli["rodStim"].setOri(self.line_orientation)
            # save current line orientation and time
            self.line_ori.append(self.line_orientation)
            self.frame_times.append(time.time())
            # show stimulus on screen
            self.display_stimuli()
            self.frame_times.append(time.time())
            self.check_response()
            # get end time of GVS
            if self.gvs_end is None:
                self.gvs_end = self._check_gvs_status("t_end_gvs", blocking=False)

        # log visual stimulus times
        line_end = time.time()
        self.logger_main.debug("{0} start visual stimulus".format(line_start))
        self.logger_main.debug("{0} stop visual stimulus".format(line_end))
        self.logger_main.info("visual stimulus duration = {0}".format(
            line_end - line_start))

        self.triggers["rodStim"] = False
        self.triggers["squareFrame"] = False
        self.display_stimuli()

    def init_trial(self):
        """
        Initialise trial
        """
        self.logger_main.debug("initialising trial {}".format(self.trial_nr))
        trial = self.trials.get_stimulus(self.trial_nr)
        self.gvs_start = None
        self.gvs_end = None
        # lists for saving the measured data
        self.line_ori = []
        self.frame_times = []
        self.line_drift = []

        # trial parameters
        self.current_mA = trial[0]
        self.line_offset = trial[1]
        self.frame_ori = trial[2]
        self.line_orientation = self.line_offset

        # stimulus asynchrony: start visual after GVS has ramped up
        self.visual_soa = self.ramp_duration_s
        self.visual_onset_delay = self.visual_soa - self.oled_delay

        # create GVS profile
        self.gvs_create.step(self.duration_s, self.current_mA)
        self.gvs_create.fade(self.f_sampling * self.ramp_duration_s)
        self.gvs_profile = self.gvs_create.stim

        # send GVS signal to handler
        self.param_queue.put(self.gvs_profile)
        self.logger_main.debug("profile sent to GVS handler")
        # check whether the gvs profile was successfully created
        if self._check_gvs_status("stim_created"):
            self.logger_main.info("gvs current profile created")
        else:
            self.logger_main.warning("WARNING: current profile not created")

    def wait_start(self):
        """
        Tell the participant to press the space bar to start the trial
        """
        self.triggers["startText"] = True
        while True:
            self.display_stimuli()
            if keyboard.is_pressed("space"):
                self.triggers["startText"] = False
                self.display_stimuli()
                break
            elif keyboard.is_pressed("esc"):
                self.quit_exp()

    def _format_data(self):
        formatted_data = "{}; {}; {}; {}; {}; {}; {}; {}; {}\n".format(
            self.trial_nr, self.current_mA, self.line_offset, self.frame_ori,
            self.gvs_start, self.gvs_end, self.line_drift, self.line_ori,
            self.frame_times)
        return formatted_data

    def run(self):
        """
        Run the experiment
        """
        for trial in range(self.n_trials):
            self.init_trial()
            self.trial_nr += 1

            # wait for space bar press to start trial
            self.wait_start()

            # send the GVS signal to the stimulator
            self.param_queue.put(True)

            # get onset time of GVS
            self.gvs_start = self._check_gvs_status("t_start_gvs")

            # draw visual line
            self.show_visual()

            # save data to file
            self.save_data.write(self._format_data())

            # self.stimulus_plot(self.visual_time, self.line_ori)
            # self.stimulus_plot(self.gvs_time, self.gvs_wave)
            # self.quit_exp()

        self.quit_exp()

    def stimulus_plot(self, xvals=None, stim=None, title=""):
        """
        Plot generated stimulus, here for debugging purposes
        """
        import matplotlib.pyplot as plt
        plt.figure()
        if xvals is not None:
            plt.plot(xvals, stim)
        else:
            plt.plot(stim)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.title(title)
        plt.show()

    def quit_exp(self):
        # send the stop signal to the GVS handler
        self.logger_main.info("quitting")
        self.param_queue.put("STOP")
        # wait for the GVS process to quit
        while True:
            if self._check_gvs_status("quit"):
                break
        # stop GVS and logging processes
        self.gvs_process.join()
        self.log_queue.put(None)
        self.log_listener.join()

        # close psychopy window and the program
        self.win.close()
        core.quit()


class SaveData:

    def __init__(self, sj, paradigm, condition, file_type="data",
                 sj_leading_zeros=0, root_dir=None):
        """
        Create a data folder and .txt or .log file, write data to file.

        :param sj: int, subject identification number
        :param paradigm: string
        :param condition: string
        :param file_type: type of file to create, either "data" (default)
        or "log" to make a log file.
        :param sj_leading_zeros: int (optional), add leading zeros to subject
        number until the length of sj_leading_zeros is reached.
        Example:
        with sj_leading_zeros=4, sj_name="2" -> sj_name="0002"
        :param root_dir: (optional) directory to place the Data folder in
        """
        # set up data folder
        if root_dir is None:
            abs_path = os.path.abspath("__file__")
            root_dir = os.path.dirname(os.path.dirname(abs_path))
        # set up subdirectory "Data" or "Log"
        assert(file_type in ["data", "log"])
        datafolder = "{}/{}".format(root_dir, file_type.capitalize())
        if not os.path.isdir(datafolder):
            os.mkdir(datafolder)

        # subject identifier with optional leading zeros
        sj_number = str(sj)
        if sj_leading_zeros > 0:
            while len(sj_number) < sj_leading_zeros:
                sj_number = "0{}".format(sj_number)

        # set up subject folder and data file
        subfolder = "{}/{}".format(datafolder, sj_number)
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        if file_type == "data":
            self.datafile = "{}/{}_{}_{}_{}.txt".format(subfolder, sj_number,
                                                        paradigm, condition,
                                                        timestr)
        else:
            self.datafile = "{}/{}_{}_{}_{}.log".format(subfolder, sj_number,
                                                        paradigm, condition,
                                                        timestr)

    def write_header(self, header):
        self.write(header)

    def write(self, data_str):
        with open(self.datafile, "a") as f:
            f.write(data_str)


class Stimuli:

    def __init__(self, window, settings_dir, n_trials=0):
        """
        Create visual stimuli with PsychoPy.

        :param window: psychopy window instance
        :param settings_dir: directory where the stimulus settings are saved
        (stimuli.json)
        :param n_trials: (optional) number of trials for on pause screen
        """
        self.stimuli = OrderedDict()
        self.triggers = {}

        self.settings_dir = settings_dir
        self.num_trials = n_trials
        self.win = window

    def create(self):
        # read stimulus settings from json file
        stim_file = "{}/stimuli.json".format(self.settings_dir)
        with open(stim_file) as json_stim:
            stim_config = json.load(json_stim)

        # cycle through stimuli
        for key, value in stim_config.items():
            # get the correct stimulus class to call from the visual module
            stim_class = getattr(visual, value.get("stimType"))
            stim_settings = value.get("settings")
            self.stimuli[key] = stim_class(self.win, **stim_settings)
            # create stimulus trigger
            self.triggers[key] = False

        return self.stimuli, self.triggers

    def draw_pause_screen(self, current_trial):
        win_width, win_height = self.win.size
        pause_screen = visual.Rect(win=self.win, width=win_width,
                                   height=win_height, lineColor=(0, 0, 0),
                                   fillColor=(0, 0, 0))
        pause_str = "PAUSE  trial {}/{} Press space to continue".format(
            current_trial, self.num_trials)
        pause_text = visual.TextStim(win=self.win, text=pause_str,
                                     pos=(0.0, 0.0), color=(-1, -1, 0.6),
                                     units="pix", height=40)
        pause_screen.draw()
        pause_text.draw()
        self.win.flip()


if __name__ == "__main__":
    exp = StepExp(sj=99, condition="test")
    exp.setup()
    exp.run()
    exp.quit_exp()
