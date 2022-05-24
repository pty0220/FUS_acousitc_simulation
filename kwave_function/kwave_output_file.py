""""""
"""
 * @file      kwave_output_file.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Lightweight wrapper around k-Wave output file.
 *
 * @date      01 June 2020, 10:00 (created)
 *            01 June 2020, 10:00 (revised)
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
from enum import Enum
import numpy as np
import h5py


class SensorSamplingType(Enum):
    RAW = 'raw'
    RMS = 'rms'
    MAX = 'max'
    MIN = 'min'


class DomainSamplingType(Enum):
    MAX = 'max_all'
    MIN = 'min_all'
    FINAL = 'final'


class KWaveOutputFile(object):
    """Represents k-Wave output file."""

    def __init__(self, file_name, reorder_data=True):
        """Constructor of k-Wave output file object.

        :param file_name:    Name of the simulation output file.
        :param reorder_data: Whether input file was created with data reordering enabled.
        """
        self.file_name = file_name
        self.reorder_data = reorder_data
        self.file_handle = None

    def open(self):
        """Open the output file."""
        self.file_handle = h5py.File(self.file_name, 'r')

    def close(self):
        """Close the output file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_pressure_everywhere(self, sampling_type: DomainSamplingType):
        """Read pressure at all point in the simulation domain (assuming it was specified to k-Wave driver).

        :param sampling_type: One of reduction methods from DomainSamplingType enum.
        :return:              Domain shaped array of pressure values loaded from the file.
        """
        dataset_name = 'p_{}'.format(sampling_type.value)
        data = np.zeros(self.file_handle[dataset_name].shape, dtype=self.file_handle[dataset_name].dtype)
        self.file_handle[dataset_name].read_direct(data)
        return self.__array_reorder(data)

    def read_pressure_at_sensor(self, sampling_type: SensorSamplingType):
        """Read pressure at sensor points/indexes (assuming it was specified to k-Wave driver).

        :param sampling_type: One of reduction methods from SensorSamplingType enum.
        :return:              Series or value (depending on reduction method) for each point/index in the sensor mask.
        """
        dataset_name = 'p_{}'.format(sampling_type.value) if sampling_type != SensorSamplingType.RAW else 'p'
        data = np.zeros(self.file_handle[dataset_name].shape, dtype=self.file_handle[dataset_name].dtype)
        self.file_handle[dataset_name].read_direct(data)
        return data

    def read_velocity_everywhere(self, sampling_type: DomainSamplingType):
        """Read velocity in each direction X,Y,Z at all point in the simulation domain.

        :param sampling_type: One of reduction methods from DomainSamplingType enum.
        :return:              Triplet (Ux, Uy, Uz) of domain shaped arrays of velocity values loaded from the file.
        """
        data_xyz = []
        for dir in ['x', 'y', 'z']:
            dataset_name = 'u{}_{}'.format(dir, sampling_type.value)
            data = np.zeros(self.file_handle[dataset_name].shape, dtype=self.file_handle[dataset_name].dtype)
            self.file_handle[dataset_name].read_direct(data)
            data_xyz.append(self.__array_reorder(data))
        return data_xyz

    def read_velocity_at_sensor(self, sampling_type: SensorSamplingType, non_staggered_raw=False):
        """Read velocity in each direction X,Y,Z at sensor points/indexes (assuming it was specified to k-Wave driver).

        :param sampling_type: One of reduction methods from SensorSamplingType enum.
        :return:              Triplet (Ux, Uy, Uz) of series or values (depending on reduction method) for each
                              point/index in the sensor mask.
        """
        if non_staggered_raw and sampling_type != SensorSamplingType.RAW:
            raise ValueError("Non-staggered velocity is available for RAW sampling type!")

        data_xyz = []
        for dir in ['x', 'y', 'z']:
            dir = '{}_non_staggered'.format(dir) if non_staggered_raw else dir
            dataset_name = 'u{}_{}'.format(dir, sampling_type.value) if sampling_type != SensorSamplingType.RAW else 'u{}'.format(dir)
            data = np.zeros(self.file_handle[dataset_name].shape, dtype=self.file_handle[dataset_name].dtype)
            self.file_handle[dataset_name].read_direct(data)
            data_xyz.append(data)
        return data_xyz

    def read_temporal_properties(self):
        """Read number and length of simulation time-steps.

        :return: Tuple of (Nt, dt).
        """
        return self.file_handle['Nt'][0], self.file_handle['dt'][0]

    def read_spatial_properties(self):
        """Read spatial properties (size and resolution) of the simulation grid

        :return: Tuple of ((Nx, Ny, Nz), (Dx, Dy, Dz))
        """
        shape = (self.file_handle['Nx'], self.file_handle['Ny'], self.file_handle['Nz'])
        delta = (self.file_handle['dx'], self.file_handle['dy'], self.file_handle['dz'])
        return self.__shape_reorder(shape), self.__shape_reorder(delta)

    def __array_reorder(self, array):
        if self.reorder_data:
            return array.transpose(2, 1, 0)
        else:
            return array.reshape(self.__shape_reorder(array.shape))

    def __shape_reorder(self, shape, pad_to=3, fill=1):
        if not self.reorder_data:
            return tuple(shape[i] for i in reversed(range(0, len(shape)))) + tuple([fill for _ in range(len(shape), pad_to)])
        else:
            return shape + tuple([fill for _ in range(len(shape), pad_to)])
