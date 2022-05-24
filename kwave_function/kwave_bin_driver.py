""""""
"""
 * @file      kwave_bin_driver.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Simple k-WAve C++/CUDA code driver in Python. This allows to generate simulation data in Python,
 *            call the binary version of k-Wave simulation code and retrieve the results in single script.
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
from typing import List

import subprocess

from kwave_function.kwave_input_file import KWaveInputFile
from kwave_function.kwave_output_file import KWaveOutputFile, SensorSamplingType, DomainSamplingType

import os
upper_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

class KWaveBinaryDriver(object):
    """Represents k-Wave solver."""
    class VerbosityLevel(Enum):
        BASIC = 0
        ADVANCED = 1
        FULL = 2

    def __init__(self, start_sampling_time=0, binary_path= upper_path +'\kwave_core\kspaceFirstOrder-CUDA.exe', reorder_data=False):
        """Constructor of k-Wave solver object.

        :param start_sampling_time: First time-step which will be sampled and store as the output.
        :param binary_path:         Path to k-Wave solver binary.
        :param reorder_data:        Whether input file was created with data reordering enabled.
        """
        self.start_sampling_time = start_sampling_time
        self.binary_path = binary_path
        self.sampling_list = {
            'pressure_at_sensor': [], 'pressure_everywhere': [],
            'velocity_at_sensor': [], 'velocity_everywhere': []
        }
        self.reorder_data = reorder_data

    def store_pressure_at_sensor(self, sampling_types: List[SensorSamplingType]):
        """Store pressure at points specified by the sensor mask/indexes in the input file using specified reduction
           methods.

        :param sampling_types: List of elements from SensorSamplingType enum.
        """
        self.sampling_list['pressure_at_sensor'].clear()
        for sampling_type in set(sampling_types):
            self.sampling_list['pressure_at_sensor'].append('p_{}'.format(sampling_type.value))

    def store_pressure_everywhere(self, sampling_types: List[DomainSamplingType]):
        """Store pressure in the whole simulation domain using specified reduction methods.

        :param sampling_types: List of reduction methods from DomainSamplingType enum.
        """
        self.sampling_list['pressure_everywhere'].clear()
        for sampling_type in set(sampling_types):
            self.sampling_list['pressure_everywhere'].append('p_{}'.format(sampling_type.value))

    def store_velocity_at_sensor(self, sampling_types: List[SensorSamplingType], non_staggered_raw=False):
        """Store velocity at points specified by the sensor mask/indexes in the input file using specified reduction
           methods.

        :param sampling_types:    List of elements from SensorSamplingType enum.
        :param non_staggered_raw: RAW non-staggered velocity is sampled when set to True.
        """
        self.sampling_list['velocity_at_sensor'].clear()
        for sampling_type in set(sampling_types):
            self.sampling_list['velocity_at_sensor'].append('u_{}'.format(sampling_type.value))

        if non_staggered_raw:
            self.sampling_list['velocity_at_sensor'].append('u_non_staggered_raw')

    def store_velocity_everywhere(self, sampling_types: List[DomainSamplingType]):
        """Store velocity in the whole simulation domain using specified reduction methods.

        :param sampling_types: List of reduction methods from DomainSamplingType enum.
        """
        self.sampling_list['velocity_everywhere'].clear()
        for sampling_type in set(sampling_types):
            self.sampling_list['velocity_everywhere'].append('u_{}'.format(sampling_type.value))

    def run(self, input_file: KWaveInputFile, output_file: KWaveOutputFile, time_steps=None):
        """Execute k-Wave solver binary with specified input file to generate specified output file.

        :param input_file:  Previously generated input file (see KWaveInputFile).
        :param output_file: Previously created output file (see KWaveOutputFile)
        :param time_steps:  Ignores number of time-steps specified in the input file when set.
        """
        exec_args = {'i': input_file.file_name, 'o': output_file.file_name}
        # if time_steps is not None:
        #     exec_args['benchmark'] = time_steps
        exec_command = self.binary_path + self.__build_exec_command(exec_args)


        proc = subprocess.Popen(exec_command)
        proc.communicate()
        returnCode = proc.poll()

        #proc = subprocess.run(exec_command, shell=True, check=True)
        #proc = subprocess.Popen(exec_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def __build_exec_command(self, key_value_args):
        exec_command = ""
        for key, value in key_value_args.items():
            exec_command += ' --{} {}'.format(key, value) if len(key) > 1 else ' -{} {}'.format(key, value)

        for sampler_type in self.sampling_list.values():
            for value in sampler_type:
                exec_command += ' --{}'.format(value) if len(value) > 1 else ' -{}'.format(value)


        return exec_command
