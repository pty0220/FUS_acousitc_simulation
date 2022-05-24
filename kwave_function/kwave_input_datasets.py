""""""
"""
 * @file      kwave_input_datasets.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Class representing database of all possible datasets in the input file.
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

from datetime import datetime


class DataSetItem(object):
    def __init__(self, size, data_type, domain_type):
        self.size = size
        self.data_type = data_type
        self.domain_type = domain_type
        self.value = None
        self.is_set = False

    def set_value(self, value):
        self.value = value
        self.size = value.shape

    def is_valid(self):
        if self.size is None:
            return False

        return not any(x is None for x in self.size)

    def __str__(self):
        return "[{0}, {1}, {2}, {3}]".format(self.size, self.data_type, self.domain_type, self.is_set)


class KWaveInputDatasets(object):
    def __init__(self, dims):
        self.dims = dims

    def get_simulation_flags(self):
        datasets = {
            'ux_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
            'uy_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
            'uz_source_flag':         DataSetItem((1, 1, 1), 'long', 'real') if self.dims > 2 else None,
            'p_source_flag':          DataSetItem((1, 1, 1), 'long', 'real'),
            'p0_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
            'transducer_source_flag': DataSetItem((1, 1, 1), 'long', 'real'),
            'nonuniform_grid_flag':   DataSetItem((1, 1, 1), 'long', 'real'),
            'nonlinear_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
            'absorbing_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
            'axisymmetric_flag':      DataSetItem((1, 1, 1), 'long', 'real'),
        }

        return KWaveInputDatasets.__filter_datasets(datasets)

    def get_grid_properties(self):
        datasets = {
            'Nx': DataSetItem((1, 1, 1), 'long',  'real'),
            'Ny': DataSetItem((1, 1, 1), 'long',  'real'),
            'Nz': DataSetItem((1, 1, 1), 'long',  'real'),
            'Nt': DataSetItem((1, 1, 1), 'long',  'real'),
            'dt': DataSetItem((1, 1, 1), 'float', 'real'),
            'dx': DataSetItem((1, 1, 1), 'float', 'real'),
            'dy': DataSetItem((1, 1, 1), 'float', 'real'),
            'dz': DataSetItem((1, 1, 1), 'float', 'real') if self.dims > 2 else None,
        }

        return KWaveInputDatasets.__filter_datasets(datasets)

    def get_medium_properties(self):
        datasets = {
            'rho0':        DataSetItem(None,      'float', 'real'),
            'rho0_sgx':    DataSetItem(None,      'float', 'real'),
            'rho0_sgy':    DataSetItem(None,      'float', 'real'),
            'rho0_sgz':    DataSetItem(None,      'float', 'real') if self.dims > 2 else None,
            'c0':          DataSetItem(None,      'float', 'real'),
            'c_ref':       DataSetItem((1, 1, 1), 'float', 'real'),

            'BonA':        DataSetItem(None,      'float', 'real'),

            'alpha_coef':  DataSetItem(None,      'float', 'real'),
            'alpha_power': DataSetItem((1, 1, 1), 'float', 'real'),
        }

        return KWaveInputDatasets.__filter_datasets(datasets)

    def get_sensor_variables(self):
        datasets = {
            'sensor_mask_type':    DataSetItem((1, 1, 1), 'long', 'real'),
            'sensor_mask_index':   DataSetItem(None,      'long', 'real'),
            'sensor_mask_corners': DataSetItem(None,      'long', 'real'),
        }

        return KWaveInputDatasets.__filter_datasets(datasets)

    def get_source_properties(self):
        datasets = {
            'u_source_mode':   DataSetItem((1, 1, 1), 'long',  'real'),
            'u_source_many':   DataSetItem((1, 1, 1), 'long',  'real'),
            'u_source_index':  DataSetItem(None,      'long',  'real'),
            'ux_source_input': DataSetItem(None,      'float', 'real'),
            'uy_source_input': DataSetItem(None,      'float', 'real'),
            'uz_source_input': DataSetItem(None,      'float', 'real') if self.dims > 2 else None,

            'p_source_mode':   DataSetItem((1, 1, 1), 'long',  'real'),
            'p_source_many':   DataSetItem((1, 1, 1), 'long',  'real'),
            'p_source_index':  DataSetItem(None,      'long',  'real'),
            'p_source_input':  DataSetItem(None,      'float', 'real'),

            'transducer_source_input': DataSetItem(None, 'float', 'real'),
            'delay_mask':              DataSetItem(None, 'long', 'real'),

            'p0_source_input':         DataSetItem(None, 'float', 'real')
        }

        return KWaveInputDatasets.__filter_datasets(datasets)

    def get_pml_variables(self):
        datasets = {
            'pml_x_size':  DataSetItem((1, 1, 1), 'long', 'real'),
            'pml_y_size':  DataSetItem((1, 1, 1), 'long', 'real'),
            'pml_z_size':  DataSetItem((1, 1, 1), 'long', 'real') if self.dims > 2 else None,

            'pml_x_alpha': DataSetItem((1, 1, 1), 'float', 'real'),
            'pml_y_alpha': DataSetItem((1, 1, 1), 'float', 'real'),
            'pml_z_alpha': DataSetItem((1, 1, 1), 'float', 'real') if self.dims > 2 else None,
        }

        return KWaveInputDatasets.__filter_datasets(datasets)

    def get_file_headers(self):
        attributes = {
            'created_by':       'Python k-Wave input generator v1.2',
            'creation_date':    str(datetime.now().isoformat()),
            'file_description': "",
            'file_type':        "input",
            'major_version':    "1",
            'minor_version':    "2"
        }

        return attributes

    @staticmethod
    def __filter_datasets(datasets):
        return {k: v for k, v in datasets.items() if v is not None}
