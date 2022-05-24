""""""
"""
 * @file      kwave_data_filters.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Helper classes used to filter data used to generate k-Wave input files.
 *
 * @date      27 May 2020, 11:00 (created)
 *            27 May 2020, 11:00 (revised)
 *
 * @copyright Copyright (C) 2012 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 *
"""

import numpy as np
from scipy.signal import blackman
from scipy.interpolate import interpn


class SpectralDataFilter(object):
    """Helper class for application of spectral filters such as smoothing."""

    @staticmethod
    def smooth(data, window_name='blackman'):
        """Smooth data using filter implemented in fourier-coefficient space using selected windowing function to
        attenuate high-frequency components.

        :param data:        Input N-dimensional array of data.
        :param window_name: Name of the windowing function to be used (only "blackman" window supported).
        :return:            Smoothed data.
        """

        data_fe = np.fft.rfftn(data, data.shape)

        if window_name == 'blackman':
            window_func = blackman
        else:
            raise ValueError('Unsupported spectrum windowing function!')

        SpectralDataFilter.__apply_window_filter(data_fe, window_func)

        return np.fft.irfftn(data_fe, data.shape)

    @staticmethod
    def __apply_window_filter(data, filter_func):
        """Apply selected windowing function to N-dimensional array of Fourier coefficients.

        :param data:        Input N-dimensional array of data.
        :param filter_func: Scalar function used to generate window coefficients in 1D "filter_func(win_length)".
        :return:            "data" multiplied by normalized window coefficients D * W^(1 / N_dims).
        """
        for axis, axis_len in enumerate(data.shape):
            filter_shape = [1, ] * data.ndim
            filter_shape[axis] = axis_len
            window_size = (2 * axis_len - 1) if axis == len(data.shape) - 1 else axis_len

            window_data = filter_func(window_size) + np.finfo(np.float32).eps
            window_data = np.fft.ifftshift(window_data)
            window_data = window_data[0:axis_len].reshape(filter_shape)
            data *= np.power(window_data, 1.0 / data.ndim)


class InterpDataFilter(object):
    """Helper class for interpolating N-dimensional arrays."""

    @staticmethod
    def staggered(data, offset):
        """Use linear interpolation to interpolate N-dimensional data to a grid shifted by "offset".

        :param data:   Input N-dimensional array of data.
        :param offset: Offset of the target grid (where abs("offset") <= 1 must hold for each dimension).
        :return:       Returns data interpolated to the offset grid. Off-grid values are copied from original data.
        """
        if len(offset) < len(data.shape):
            offset = offset + tuple([0 for _ in range(len(offset), len(data.shape))])
        else:
            offset = offset[0:len(data.shape)]

        if any(map(lambda o: abs(o) > 1, offset)):
            raise ValueError("Invalid offset specified (must be less than 1)!")

        grid = tuple(map(np.arange, data.shape))
        mesh = np.array(np.meshgrid(*tuple(map(lambda d: d[0] + d[1], zip(grid, offset))), indexing='ij'))
        mesh = np.moveaxis(mesh, 0, len(data.shape)).reshape(data.shape + (len(data.shape),))

        staggered_data = interpn(grid, data, mesh, bounds_error=False, fill_value=float('nan'))

        for i in range(0, len(data.shape)):
            if offset[i] == 0:
                continue

            s = [slice(None) for _ in range(0, len(data.shape))]
            s[i] = slice(-1, None) if offset[i] > 0 else slice(0, 1)

            staggered_data[tuple(s)] = data[tuple(s)]

        return staggered_data
