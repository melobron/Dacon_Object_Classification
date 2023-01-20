import os
import numpy as np
import scipy
from pathlib import Path
from utils import utils


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


class Stitcher(object):
    def __init__(self, args):
        self.cached_2d_windows = dict()
        self.file_list = dict()
        self.args = args

        self.img_list = ['ri', 'nuc', 'oli', 'mito', 'act', 'mem']
        self.window_2d = self._window_2D(args.patch_size).transpose(2, 0, 1)

    def _update(self, img_name, coord, **imgs):
        patch = {coord:imgs}

        if img_name not in self.file_list:
            self.file_list.update({img_name:patch})
        else:
            self.file_list[img_name].update(patch)

        self._check_for_saving()

    def _check_for_saving(self):
        full_index = ((512 * self.args.zoom_factor - 128) / 64 + 1) ** 2
        for k, v in self.file_list.items():
            if len(v) == full_index:
                self._save(k, v)
                self.file_list.pop(k)
                break

    def _save(self, k, v):
        save_path = Path(self.args.img_save_path) / self.args.inifile / k
        print(save_path)

        size = [64, int(self.args.zoom_factor*512), int(self.args.zoom_factor*512)]
        temp = {i:np.zeros(size) for i in self.img_list}

        for _k, _v in v.items():
            y, x = _k.split('_')
            y, x = int(y), int(x)
            for i in self.img_list:
                temp[i][:, y:y+self.args.patch_size, x:x+self.args.patch_size] += _v[i] * self.window_2d

        for i in self.img_list:
            temp[i] /= 3

        utils.image_save(str(save_path), **temp)

    def print(self):
        for k, v in self.file_list.items():
            print(k)
            print(v.keys())
            for _k, _v in v.items():
                print(_k, _v['ri'].shape, end=', ')
            print()

    def _window_2D(self, window_size=128, power=2):
        """
            Make a 1D window function, then infer and return a 2D window function.
            Done with an augmentation, and self multiplication with its transpose.
            Could be generalized to more dimensions.
            """
        key = "{}_{}".format(window_size, power)
        if key in self.cached_2d_windows:
            wind = self.cached_2d_windows[key]
        else:
            wind = _spline_window(window_size, power)
            wind = np.expand_dims(np.expand_dims(wind, 3), 3)
            wind = wind * wind.transpose(1, 0, 2)
            self.cached_2d_windows[key] = wind
        return wind
