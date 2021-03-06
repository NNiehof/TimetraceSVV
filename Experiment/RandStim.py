from random import shuffle
import numpy as np


class RandStim:

    def __init__(self, current_mA=None, line_offset=None, frame_angles=None):
        """
        Object that creates a randomised list of trials, out of a range of
        stimulus values and conditions.
        """
        self.trial_list = []
        for curr in current_mA:
            for line_mu in line_offset:
                # add conditions with a visual frame and 2 mA current
                if curr == 2.0:
                    for frame in frame_angles:
                        self.trial_list.append([curr, line_mu, frame])
                else:
                    self.trial_list.append([curr, line_mu, None])
        shuffle(self.trial_list)

    def get_stimulus(self, trial_nr):
        """
        Return stimulus and conditions for next trial
        :return trial: list with conditions
        """
        return self.trial_list[trial_nr]

    def get_n_trials(self):
        """
        Return number of trials in trial list.
        """
        return len(self.trial_list)


if __name__ == "__main__":
    cond = {"current_mA": [0.5, 3.0, 1.5], "line_offset": [1.0, 20.0, 30.0]}
    s = RandStim(**cond)
    print(s.trial_list)
    print(np.shape(s.trial_list))
