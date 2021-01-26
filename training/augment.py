from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline
import warnings

"""
Augmentation techniques:
    1. Scaling
    2. Rotation
    3. Cropping + interpolation
    4. Magnitude warp
    5. Time warp
    6. Jittering
    7. Permutation
(usage example at the end of this file)
"""


class Augmenter:
    def random_augment(self, data):
        pass


class NumpyAugmenter(Augmenter):
    # type = np.float32

    def __init__(self,
                 input_shape: Union[list, tuple],
                 max_num_transformations: int = 99,
                 shuffle_transformations: bool = False,

                 scale_range: Union[list, tuple] = None,
                 rotate_x_range: Union[list, tuple] = None,
                 rotate_y_range: Union[list, tuple] = None,
                 rotate_z_range: Union[list, tuple] = None,
                 interp_cut_range: Union[list, tuple] = None,
                 magnitude_warp_sigma_range: Union[list, tuple] = None,
                 nums_magnitude_warp_knots: Union[list, tuple] = None,
                 time_warp_sigma_range: Union[list, tuple] = None,
                 nums_time_warp_knots: Union[list, tuple] = None,
                 noise_jitter_factor_ranges: Union[list, tuple] = None,
                 nums_permute_parts: Union[list, tuple] = None
                 ) -> None:
        """

        :param input_shape: input window shape,
            e.g. [2500,6]
        :param augmentation_apply_rate: from 0 to 1,
            independent augment rate of each module,
            e.g. 1.
        :param scale_range: list of 2 positive float,
            the range of scaling rate by which signals are scaled up and down,
            e.g. scale_range = [0.1, 0.2],
                signal will be multiplied with 0.8~0.9 or 1.1~1.2
        :param rotate_x_range, rotate_y_range, rotate_z_range:
            list of 2 positive floats, range of degree to rotate signals,
            e.g. [10., 20.]
                signal will be rotated by 10~20 degs or -20~-10 degs
        :param interp_cut_range: list of 2 positive floats,
            range of how much of a
            window to be cropped before interpolating,
            e.g. [0.1, 0.25]
        :param magnitude_warp_sigma_range: list of 2 positive floats,
            range of standard deviation of Gaussian noise of magnitude curve,
            e.g. [0.1, 0.15]
        :param nums_magnitude_warp_knots: list of ints,
            number of knots in a magnitude curve,
            e.g. [4,5,6]
        :param time_warp_sigma_range: list of 2 positive floats,
            range of standard deviation of Gaussian noise of temporal curve,
            e.g. [0.1, 0.2]
        :param nums_time_warp_knots: list of ints,
            number of knots in a temporal curve,
            e.g. [3,4,5]
        :param noise_jitter_factor_ranges:
            list of 2 lists of 2 positive floats corresponding to each modality,
            standard deviation of Gaussian noise added to signals,
            e.g. [[2e-2, 2.5e-2], [9e-2, 9.5e-2]]
        :param nums_permute_parts: list of positive ints,
            number of segments to be divided and swapped randomly
            e.g. [2,3,4,5]
        """

        self.input_len = input_shape[0]
        self.input_channel = input_shape[-1]
        self.num_modals = self.input_channel // 3
        self.max_num_transformations = max(max_num_transformations, 0)
        self.shuffle_transformations = shuffle_transformations

        if self.input_channel % 3 != 0:
            warnings.warn('input data is not triaxial')

        if noise_jitter_factor_ranges is not None and len(
                noise_jitter_factor_ranges) != self.num_modals:
            raise ValueError(
                'number of noise jitter factors must be '
                'equal to the number of tri-axial modalities')

        # init list augmentation method (function)
        self.list_aug_func = []

        # if parameter(s) of a augmentation technique is specified,
        # define its parameters and add the method to the above list.

        if scale_range is not None:
            if scale_range[-1] > 0.2:
                warnings.warn('scale_rate may be too large.')
            self.scale_range = np.array(scale_range)
            self.list_aug_func.append(self.apply_scaling)

        if (rotate_x_range is not None) and (rotate_y_range is not None) and (
                rotate_z_range is not None):
            # convert angles from angle to radian
            self.rotate_x_range = np.array(rotate_x_range) / 180. * np.pi
            self.rotate_y_range = np.array(rotate_y_range) / 180. * np.pi
            self.rotate_z_range = np.array(rotate_z_range) / 180. * np.pi
            self.list_aug_func.append(self.apply_rotation)

        if interp_cut_range is not None:
            if interp_cut_range[1] >= 0.25:
                warnings.warn('interp_max_cut_rate may be too large.')
            self.interp_cut_range = np.array(interp_cut_range) / 2.
            self.list_aug_func.append(self.apply_crop_interpolation)

        if (magnitude_warp_sigma_range is not None) and (
                nums_magnitude_warp_knots is not None):
            self.magnitude_warp_sigma_range = magnitude_warp_sigma_range
            self.nums_magnitude_warp_knots = nums_magnitude_warp_knots
            self.list_aug_func.append(self.apply_magnitude_warp)

        if (time_warp_sigma_range is not None) and (
                nums_time_warp_knots is not None):
            self.time_warp_sigma_range = time_warp_sigma_range
            self.nums_time_warp_knots = nums_time_warp_knots
            self.list_aug_func.append(self.apply_time_warp)

        if noise_jitter_factor_ranges is not None:
            self.noise_jitter_factor_ranges = noise_jitter_factor_ranges
            self.list_aug_func.append(self.apply_jittering)

        if nums_permute_parts is not None:
            """
            Pre-define start and stop indices for permutation.
            self.list_permute_segment_index is a list of 2d-arrays,
            each array is of a corresponding number of permute parts.
            Example of an array containing start and stop indices for
            3 permute parts, with window size is 1500:
            [[0,500],
             [500,1000],
             [1000,1500]]
            """

            self.list_permute_segment_index = []
            for n in nums_permute_parts:
                if n != 1:
                    index = np.linspace(0, self.input_len, n + 1, dtype=int)
                    index = np.array([index[:-1], index[1:]]).T
                    self.list_permute_segment_index.append(index)
            if len(self.list_permute_segment_index) > 0:
                self.list_permute_choice_idx = np.arange(
                    len(self.list_permute_segment_index)
                )
                self.list_aug_func.append(self.apply_permutation)

        if len(self.list_aug_func) == 0:
            warnings.warn(
                'Augmenter will not be used because '
                'all initial params are None!')

        self.max_num_transformations = min(self.max_num_transformations,
                                           len(self.list_aug_func))
        self.list_aug_func = np.array(self.list_aug_func)

    def random_augment(self, orgdata):
        """
        Apply augmentation methods in self.list_aug_func
        :param orgdata:
            shape (timestep, channel) channel must be divisible by 3,
            otherwise bugs may occur
        :return: shape (timestep, channel)
        """

        # check input shape
        if orgdata.shape[0] != self.input_len \
                or orgdata.shape[1] != self.input_channel:
            raise ValueError(
                f'wrong input shape, expect ({self.input_len}, '
                f'{self.input_channel}) but get {orgdata.shape}')

        # clone data array to avoid making changes to original data array
        data = np.copy(orgdata)

        # not shuffle and not choose
        if (not self.shuffle_transformations) \
                and (self.max_num_transformations == len(self.list_aug_func)):
            list_aug_func = self.list_aug_func

        # if shuffle, and choose or not choose
        elif self.shuffle_transformations:
            list_aug_func = np.random.choice(
                self.list_aug_func,
                size=self.max_num_transformations,
                replace=False  # not duplicate func
            )

        # if choose, and not shuffle
        else:
            choose_idx = np.sort(np.random.choice(
                np.arange(len(self.list_aug_func)),
                size=self.max_num_transformations,
                replace=False
            ))
            list_aug_func = self.list_aug_func[choose_idx]

        # apply methods in self.list_aug_func
        for func in list_aug_func:
            data = func(data)

        return data

    """ <<<<<< module: SCALING >>>>>> """

    def apply_scaling(self, data):
        range_ = \
            1 \
            + np.random.choice([-1, 1]) \
            * np.random.uniform(
                self.scale_range[0],
                self.scale_range[1],
                size=[1, 3]
            )  # shape (1, 3)

        # create scaling factor, group by x/y/z axis
        frame = np.dot(
            np.ones([self.input_len, 1]),  # shape (n, 1)
            range_
        )  # shape (n, 3)

        # aplly scaling
        if self.num_modals != 1:
            frame = np.hstack([frame] * self.num_modals)  # shape (n, 6)

        return data * frame

    """ <<<<<< module: ROTATION >>>>>> """

    def apply_rotation(self, data):
        """
        Generate random angles and apply rotation.
        """
        rotate_angles = np.array([
            np.random.uniform(*self.rotate_x_range),
            np.random.uniform(*self.rotate_y_range),
            np.random.uniform(*self.rotate_z_range)
        ]) * np.random.choice([-1, 1], size=3)

        for i in range(0, self.input_channel, 3):
            data[:, i:i + 3] = self.rotate(data[:, i:i + 3],
                                           *rotate_angles)

        return data

    def rotate(self, data, rotate_x, rotate_y, rotate_z):
        """
        Rotate an array
        :param data: shape (timestep, 3)
        :param rotate_x, rotate_y, rotate_z: angle in RADIAN
        :return: shape (timestep, 3)
        """

        # create rotation filters
        rotate_x = np.array([
            [1, 0, 0],
            [0, np.cos(rotate_x), -np.sin(rotate_x)],
            [0, np.sin(rotate_x), np.cos(rotate_x)]
        ])
        rotate_y = np.array([
            [np.cos(rotate_y), 0, np.sin(rotate_y)],
            [0, 1, 0],
            [-np.sin(rotate_y), 0, np.cos(rotate_y)]
        ])
        rotate_z = np.array([
            [np.cos(rotate_z), -np.sin(rotate_z), 0],
            [np.sin(rotate_z), np.cos(rotate_z), 0],
            [0, 0, 1]
        ])

        # rotate original data by multiply it with rotation filters
        rotate_filters = np.dot(np.dot(rotate_x, rotate_y), rotate_z)
        data = np.dot(rotate_filters, data.T).T
        return data

    """ <<<<<< module: CROPPING AND INTERPOLATION >>>>>> """

    def apply_crop_interpolation(self, data):
        """
         Generate start point and stop point randomly and crop data,
         then apply interpolation.
        """
        random_ = np.random.uniform(*self.interp_cut_range, size=2)
        start_point = int(self.input_len * random_[0])
        stop_point = int(self.input_len * (1 - random_[1]))
        data = data[start_point:stop_point]

        data = self.interpolation_2d(data, self.input_len)
        return data

    def npinterp(self, source, length):
        """
        Interpolation for 1D array.
        :param source: 1-D vector (len,)
        :param length: expected length after interpolation
        :return: augmented data array
        """
        return np.interp(np.linspace(0, len(source) - 1, length),
                         np.arange(len(source)), source)

    def interpolation_2d(self, source, length):
        """
        Interpolation for 2D array
        (apply 1D interpolation along the channel axis).
        :param source: data array shape (len, channel)
        :param length: expected length after interpolation
        :return: (len, channel)
        """

        # placeholder for result data
        new_source = np.empty([length, source.shape[-1]], dtype=np.float32)
        # apply 1D interpolation along the channel axis
        for i in range(source.shape[-1]):
            new_source[:, i] = self.npinterp(source[:, i], length)
        return new_source

    """ <<<<<< module: MAGNITUDE WARP >>>>>> """

    def apply_magnitude_warp(self, data):
        """
        Apply mag warp along the channel axis (group by x/y/z channels)
        by creating n_channels random curves and
        multiply them with original data.
        """
        # randomly pick a number of knots to create random curves
        knot = np.random.choice(self.nums_magnitude_warp_knots)

        # generate 3 curves for 3 axes
        sigma = np.random.uniform(*self.magnitude_warp_sigma_range)
        curve = self.gen_random_curve(3, sigma, knot)

        # stack generated curves, group by x/y/z channels
        curve = np.hstack([curve] * self.num_modals)

        # multiply curves with original data
        return data * curve

    def gen_random_curve(self, num_curve, sigma, knot):
        """
        Generate random curve(s) for Magnitude warp and Time warp modules.
        :param num_curve: number of curves to generate
        :param sigma: standard deviation of generated curves
        :param knot: number of knots in a curve
        :return: array shape [curve length, num curve]
        """
        xx = (np.ones((num_curve, 1)) * (
            np.arange(0, self.input_len, (self.input_len - 1) / (knot + 1)))
              ).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curve))
        x_range = np.arange(self.input_len)
        results = []
        for i in range(num_curve):
            results.append(CubicSpline(xx[:, i], yy[:, i])(x_range))
        results = np.array(results).T
        return results

    """ <<<<<< module: TIME WARP >>>>>> """

    def apply_time_warp(self, data):
        """
        Apply time warp along the channel axis
        by creating a random curve to distort the timestream of data array.
        """

        # randomly pick a number of knots to create a random curve
        knot = np.random.choice(self.nums_time_warp_knots)

        # create distorted timestamps array
        sigma = np.random.uniform(*self.time_warp_sigma_range)
        tt_new = self.distort_timestep(sigma=sigma, knot=knot)

        # create placeholder for augmented data
        data_new = np.empty([self.input_len, self.input_channel])

        # apply new timestamp on original data array
        data_range = np.arange(self.input_len)
        for i in range(self.input_channel):
            data_new[:, i] = np.interp(data_range, tt_new, data[:, i])
        return data_new

    def distort_timestep(self, sigma=0.2, knot=4):
        """
        Create distorted timestream array.
        :param sigma: standard deviation of random curve
        :param knot: number of knots in a curve
        :return: distorted timestamps array shape [length, ]
        """

        # generate a random curve shape [length, ]
        tt = self.gen_random_curve(1, sigma, knot)[:, 0]
        # calculate cumulative sum of random curve to get distorted timestream
        tt_cum = np.cumsum(tt)
        # re-scale the distorted timestream to be in original range
        t_scale = (self.input_len - 1) / tt_cum[-1]
        tt_cum *= t_scale
        return tt_cum

    """ <<<<<< module: JITTERING >>>>>> """

    def apply_jittering(self, data):
        """
        Add gaussian noise to data array,
        group by accelerometer and gyroscope signals
        """
        for modal_index, axis_index in enumerate(
                range(0, self.input_channel, 3)):
            scale = np.random.uniform(
                *self.noise_jitter_factor_ranges[modal_index])

            data[:, axis_index:axis_index + 3] = \
                data[:, axis_index:axis_index + 3] + \
                np.random.normal(loc=0,
                                 scale=scale,
                                 size=[self.input_len, 3])
        return data

    """ <<<<<< module: PERMUTATION >>>>>> """

    def apply_permutation(self, data):
        """
        Randomly split data array into multiple parts along temporal axis,
        then randomly swap parts.
        """
        # randomly pick a number of parts (pick its start/stop indices array)
        index = np.random.choice(self.list_permute_choice_idx)
        index = self.list_permute_segment_index[index]
        # randomly swap parts' indices
        index = np.random.permutation(index)
        # create placeholder for augmented data
        perm_data = np.empty([self.input_len, self.input_channel],
                             dtype=data.dtype)
        # put swapped parts into the placeholder
        marker = 0
        for segment_index in index:
            this_len = segment_index[1] - segment_index[0]
            perm_data[marker:marker + this_len] = data[segment_index[0]:
                                                       segment_index[1]]
            marker += this_len
        return perm_data
