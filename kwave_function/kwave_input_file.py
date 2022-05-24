""""""
"""
 * @file      kwave_input_file.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Class representing k-Wave input file. After the file is created using constructor and "open()" call,
 *            the data can be directly writen using using "write_*" methods. When data is written the file *HAS* to be
 *            closed using "close()" call at which point it is finalized and closed.
 *            Typical approach is to use KWaveInputFile together with "with as" construct:
 *            with KWaveInputFile(...) as input_file:
 *                input_file.write_*(...)
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
import operator
from enum import Enum
from itertools import accumulate
from functools import reduce
import numpy as np
import h5py
from kwave_function.kwave_input_datasets import KWaveInputDatasets
from kwave_function.kwave_data_filters import InterpDataFilter


class KWaveInputFile(object):
    """Represents k-Wave input file."""

    class SourceMode(Enum):
        DIRICHLET = 0
        ADDITIVE = 1

    class SensorType(Enum):
        INDEX = 0
        CORNERS = 1

    def __init__(self, file_name, domain_shape, nt, spatial_delta, time_delta, c_ref=1500.0, pml_alpha=2, pml_size=20,
                 reoder_output_data=True):
        """Constructor of k-Wave input file generator.

        :param file_name:          Name of the generated input file.
        :param domain_shape:       Spatial shape of the simulation domain in number of grid points.
        :param nt:                 Total number of time-steps of "time_delta" to be performed in the simulation.
        :param spatial_delta:      Spatial resolution of the simulation domain in meters [m].
        :param time_delta:         Temporal resolution of the simulation in seconds [s].
        :param c_ref:              Reference sound-speed of the medium used to compute k-space correction coefficient [m/s].
        :param pml_alpha:          Absorption coefficient [Nepers per grid point]
        :param pml_size:           Width of the PML used to absorb wave leaving the simulation domain [grid points].
        :param reoder_output_data: When set to True simulation data are reordered so that MATLAB semantics is preserved.
                                   Otherwise *ONLY* the simulation shape is changed to (Nz, Ny, Nx) and data ordering
                                   is preserved (spatial resolution, pml properties etc. are likewise reordered)
        """
        self.file_name = file_name
        self.domain_shape = domain_shape
        self.dims = len(domain_shape)
        self.nt = nt
        self.spatial_delta = spatial_delta
        self.time_delta = time_delta
        self.c_ref = c_ref
        self.pml_alpha = self.__make_nd_tuple(pml_alpha, self.dims)
        self.pml_size = self.__make_nd_tuple(pml_size, self.dims)
        self.reoder_output_data = reoder_output_data

        self.file_handle = None
        self.file_state = None
        self.data_set_db = KWaveInputDatasets(self.dims)

    def open(self):
        """Creates output HDF5 file.
        """
        self.file_handle = h5py.File(self.file_name, 'w')
        self.__initialize_file_state()

    def close(self):
        """Finalize the output file and close it.
        """
        if self.file_handle:
            self.__finalize_and_validate_file()
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_medium_density(self, rho0):
        """Write medium density [kg/m^3] into the opened file. The value has to be either scalar or N-dim array of same
        shape as the simulation domain.

        :param rho0: Medium density N-dim array or scalar float value [kg/m^3].
        """
        self.__update_file_state(self.data_set_db.get_medium_properties())

        if isinstance(rho0, (np.ndarray,)):
            self.__write_global_field('rho0', rho0)
            self.__write_global_field('rho0_sgx', InterpDataFilter.staggered(rho0, (0.5, 0.0, 0.0)))
            self.__write_global_field('rho0_sgy', InterpDataFilter.staggered(rho0, (0.0, 0.5, 0.0)))
            self.__write_global_field('rho0_sgz', InterpDataFilter.staggered(rho0, (0.0, 0.0, 0.5)))
        else:
            self.__write_scalar('rho0', rho0)
            self.__write_scalar('rho0_sgx', rho0)
            self.__write_scalar('rho0_sgy', rho0)
            self.__write_scalar('rho0_sgz', rho0)

    def write_medium_sound_speed(self, c0, c_ref=None):
        """Write medium sound speed [m/s] into the opened file. The value has to be either scalar or N-dim array of the
        same shape as the simulation domain.

        :param c0:    Medium sound speed N-dim array or scalar float value [m/s].
        :param c_ref: Reference sound-speed of the medium used to compute k-space correction coefficient [m/s].
        """
        self.__update_file_state(self.data_set_db.get_medium_properties())

        if isinstance(c0, (np.ndarray,)):
            self.__write_global_field('c0', c0)
        else:
            self.__write_scalar('c0', c0)

        self.c_ref = c_ref or self.c_ref
        self.__write_scalar('c_ref', self.c_ref)

    def write_medium_non_linear(self, b_on_a):
        """Write non-linear coefficient (B/A) into the opened file. The value has to be either scalar or N-dim array of the
        same shape as the simulation domain.

        :param b_on_a: Medium non-linear coefficient (B/A) N-dim array or scalar float.
        """
        self.__update_file_state(self.data_set_db.get_medium_properties())

        if isinstance(b_on_a, (np.ndarray,)):
            self.__write_global_field('BonA', b_on_a)
        else:
            self.__write_scalar('BonA', b_on_a)

    def write_medium_absorbing(self, alpha_coeff, alpha_power=1.0):
        """Write frequency dependent absorption properties of the medium.

        :param alpha_coeff: Power law absorption coefficient (N-dim array or scalar float).
        :param alpha_power: Power law absorption exponent (scalar float).
        """
        self.__update_file_state(self.data_set_db.get_medium_properties())

        if isinstance(alpha_coeff, (np.ndarray,)):
            self.__write_global_field('alpha_coeff', alpha_coeff)
        else:
            self.__write_scalar('alpha_coeff', alpha_coeff)

        self.__write_scalar('alpha_power', alpha_power)

    def write_source_input_p0(self, p0):
        """Write initial acoustic pressure distribution it the simulation domain.

        :param p0: Initial pressure (N-dim array) in [Pa].
        """
        self.__update_file_state(self.data_set_db.get_source_properties())
        self.__write_global_field('p0_source_input', p0)

    def write_source_input_p(self, index, data, mode: SourceMode, c0):
        """Write pressure source data to the output file. Pressure sources support two modes:

        Single series mode (SSM): This mode uses single series to drive all source points.

        Multi series mode (MSM):  This mode uses separate series to drive each source point.

        Sources can be either dirichlet or additive in both modes.

        :param index: Zero based indexes into the simulation domain denoting source grid points.
        :param data:  Source values to be applied at each time step [Pa]. In SSM 1D vector is expected while in MSM
                      2D matrix of (len(index), [signal_len]) is expected instead.
        :param mode:  Dirichlet or Additive source mode (member of SourceMode enum)
        :param c0:    Medium sound speed used to normalize the source (N-dim array or scalar float value [m/s]).
                      Reference sound speed is used if not provided.
        """
        self.__update_file_state(self.data_set_db.get_source_properties())

        many = len(data.shape) > 1
        self.__write_scalar('p_source_mode', mode.value)
        self.__write_scalar('p_source_many', 1 if many else 0)
        self.__write_field('p_source_index', index.reshape((index.shape[0], 1, 1)) + 1)

        if np.isscalar(c0):
            norm_c0 = c0
        elif isinstance(c0, np.ndarray):
            c0_t = self.__array_to_output_order(c0)
            if many:
                norm_c0 = c0_t.flat[index].reshape(self.__shape_to_output_order((index.size, 1), 2))
            else:
                norm_c0 = c0_t.flat[index[0]]
        else:
            norm_c0 = self.c_ref

        if mode == KWaveInputFile.SourceMode.DIRICHLET:
            p_norm_coeff = 1.0 / (self.dims * norm_c0 * norm_c0)
        else:
            p_norm_coeff = (2.0 * self.time_delta) / (self.dims * norm_c0 * self.spatial_delta[0])

        data_t = (data * p_norm_coeff).reshape(self.__shape_to_output_order(data.shape))

        if not many:
            self.__write_field('p_source_input', data_t.reshape((1, data.shape[0], 1)))
        else:
            if data_t.shape[0] != index.shape[0]:
                raise ValueError("One source series has to be specified for each source index!")

            self.__write_field('p_source_input', data_t)

    def write_source_input_u(self, index, data_ux, data_uy, data_uz, mode: SourceMode, c0):
        """Write velocity source data to the output file. Velocity source consist of vector valued elements for each
        time step. While its possible to leave any of vector components implicitly zero at least one component has to
        be specified. Velocity sources support two modes:

        Single series mode (SSM): This mode uses single series to drive all source points.

        Multi series mode (MSM):  This mode uses separate series to drive each source point.

        The data series "data_u{x,y,z}" are expected to be 1D vectors in the SSM mode and 2D matrices of
        (len(index), [signal_len]) in MSM mode.
        Sources can be either dirichlet or additive in both modes.

        :param index:   Zero based indexes into the simulation domain denoting source grid points.
        :param data_ux: X-axis component of source values applied at each time step.
        :param data_uy: Y-axis component of source values applied at each time step.
        :param data_uz: Z-axis component of source values applied at each time step.
        :param mode:    Dirichlet or Additive source mode (member of SourceMode enum)
        :param c0:      Medium sound speed used to normalize the source (N-dim array or scalar float value [m/s]).
                        Reference sound speed is used if not provided.
        """
        self.__update_file_state(self.data_set_db.get_source_properties())

        # Check if provided data match SSM or MSM mode.
        data_u_xyz_shapes = [d.shape for d in [data_ux, data_uy, data_uz] if d is not None]
        many = any(map(lambda s: len(s) > 1, data_u_xyz_shapes))

        # Check if present data series are of matching shape.
        if len(set(data_u_xyz_shapes)) != 1:
            raise ValueError("Velocity source data size mismatch (all data_{x,y,z} must be of the same size if present)!")

        self.__write_scalar('u_source_mode', mode.value)
        self.__write_scalar('u_source_many', 1 if many else 0)
        self.__write_field('u_source_index', index.reshape((index.shape[0], 1, 1)) + 1)

        if np.isscalar(c0):
            norm_c0 = c0
        elif isinstance(c0, np.ndarray):
            c0_t = self.__array_to_output_order(c0)
            if many:
                norm_c0 = c0_t.flat[index].reshape(self.__shape_to_output_order((index.size, 1), 2))
            else:
                norm_c0 = c0_t.flat[index[0]]
        else:
            norm_c0 = self.c_ref

        u_norm_coeff = 2.0 * norm_c0 * self.time_delta

        # Write all present data series to the file.
        for name, data, sd in zip(['ux_source_input', 'uy_source_input', 'uz_source_input'], [data_ux, data_uy, data_uz], self.spatial_delta):
            if data is None:
                continue

            u_dir_norm_coeff = u_norm_coeff / sd if mode != KWaveInputFile.SourceMode.DIRICHLET else 1.0
            data_t = (data * u_dir_norm_coeff).reshape(self.__shape_to_output_order(data.shape))

            if not many:
                self.__write_field(name, data_t.reshape(1, data.shape[0], 1))
            else:
                if data.shape[0] != index.shape[0]:
                    raise ValueError("One source series has to be specified for each source index!")
                self.__write_field(name, data_t)

    def write_source_input_transducer(self, index, series_data, delay_mask, c0):
        """Write transducer source data to the output file. Transducer source is additive velocity source, which is flat
        and assumes that all the energy is transferred in the X-direction. While signals in all source points are the
        same each point is associated with "delay_mask", which is added to current time step to compute offset in the
        "series_data".

        :param index:       Zero based indexes into the simulation domain denoting source grid points (1D vector).
        :param series_data: X-axis component of source values applied at each time step (1D vector).
        :param delay_mask:  Delay (# of time steps) of the series for each source point in "index" (1D vector).
        :param c0:          Medium sound speed used to normalize the source (N-dim array or scalar float value [m/s]).
                            Reference sound speed is used if not provided.
        """
        self.__update_file_state(self.data_set_db.get_source_properties())

        if any(map(lambda d: len(d.shape) != 1, [index, series_data, delay_mask])):
            raise ValueError("All transducer input data have to be 1D vectors.")

        if index.shape != delay_mask.shape:
            raise ValueError("Index and Delay mask length mismatch!")

        if self.file_state['u_source_index'].is_set and self.file_state['u_source_mode'].is_set:
            raise ValueError("Transducer source and Velocity sources cannot be used at the same time!")

        if np.isscalar(c0):
            norm_c0 = c0
        elif isinstance(c0, np.ndarray):
            c0_t = self.__array_to_output_order(c0)
            norm_c0 = np.mean(c0_t.flat[index])
        else:
            norm_c0 = self.c_ref

        norm_coeff = 2.0 * norm_c0 * self.time_delta / self.spatial_delta[0]

        self.__write_field('u_source_index', index.reshape((index.shape[0], 1, 1)) + 1)
        self.__write_field('transducer_source_input', (series_data * norm_coeff).reshape((series_data.shape[0], 1, 1)))
        self.__write_field('delay_mask', delay_mask.reshape((delay_mask.shape[0], 1, 1)))

    def write_sensor_mask_index(self, index):
        """Write index sensor mask to the output file. Sensor mask denotes grid points whose value will be stored to the
        output file at each time step.

        :param index: Zero based indexes into the simulation domain denoting sensor grid points (1D vector).
        """
        self.__update_file_state(self.data_set_db.get_sensor_variables())

        if len(index.shape) != 1:
            raise ValueError("Sensor mask index is expected to be 1D vector!")

        self.__write_scalar('sensor_mask_type', KWaveInputFile.SensorType.INDEX.value)
        self.__write_field('sensor_mask_index', index.reshape((index.shape[0], 1, 1)) + 1)

    def write_sensor_mask_corners(self, corners):
        """Write corners sensor mask to the output file. Corners sensor mask denotes cuboids in the simulation domain
        whose values will be stored at each time step.

        :param corners: Corners of sampled axis-aligned cuboids stored as 3D array of: ((LTx, LTy, LTz), (RBx, RBy, RBz)), ... in
                        Nx2x3 array. Example: [[(0, 0, 0), (10, 10, 10)], [(20, 20, 20), (30, 30, 30)]] is two cuboids,
                        where first starts at grid point 0,0,0 and extends to 10,10,10 and second starts at 20,20,20 and
                        ends at 30,30,30.
        """
        self.__update_file_state(self.data_set_db.get_sensor_variables())

        if len(corners.shape) != 3 or corners.shape[2] != 3 or corners.shape[1] != 2:
            raise ValueError("Corners mask is expected to be of shape Nx2x3!")

        self.__write_scalar('sensor_mask_type', KWaveInputFile.SensorType.CORNERS.value)
        self.__write_field('sensor_mask_corners', corners.reshape((corners.shape[0], 2*3, 1)))

    def domain_mask_to_index(self, mask):
        """Convert the domain shaped non-zero mask to flat array of indexes in the data order written to the file.
        This should be used to generate indexes for source and sensor masks.

        :param mask: Domain shaped ND-array representing the non-zero mask.
        :return:     Flat array of indexes valid in output file data order.
        """
        if mask.shape != self.domain_shape:
            raise ValueError("Mask shape has to exactly match the domain shape!")

        mask_t = self.__array_to_output_order(mask)
        return np.flatnonzero(mask_t)

    def domain_mask_values_in_index_order(self, mask):
        """Extract non-zero values from domain shaped mask to flat array of values. The order of values is returned in
        the order in which indexes would be written to the file.

        :param mask: Domain shaped ND-array representing the non-zero mask.
        :return:     Flat array of non-zero values in output file data order.
        """
        if mask.shape != self.domain_shape:
            raise ValueError("Mask shape has to exactly match the domain shape!")

        mask_t = self.__array_to_output_order(mask)
        return mask_t.flat[np.flatnonzero(mask_t)]

    def __array_to_output_order(self, array):
        if self.reoder_output_data:
            return array.transpose(2, 1, 0)
        else:
            return array.reshape(self.__shape_to_output_order(array.shape))

    def __shape_to_output_order(self, shape, pad_to=3, fill=1):
        if not self.reoder_output_data:
            return tuple(shape[i] for i in reversed(range(0, len(shape)))) + tuple([fill for _ in range(len(shape), pad_to)])
        else:
            return shape + tuple([fill for _ in range(len(shape), pad_to)])

    # def __domain_index_to_transpose_order(self, index):
    #     shape = KWaveInputFile.__make_nd_tuple(self.domain_shape, 3)
    #     shape_t = (shape[2], shape[1], shape[0])
    #     coeffs_t = tuple(accumulate((1,) + shape_t, operator.mul))
    #     z, y, x = tuple([(index // c) % s for s, c in zip(shape_t, coeffs_t)])
    #
    #     coeffs = tuple(accumulate((1,) + shape, operator.mul))
    #     return np.sort(reduce(operator.add, [a * c for a, c in zip((x, y, z), coeffs)]))

    def __write_scalar(self, name, value):
        if name not in self.file_state:
            return

        item = self.file_state[name]
        item.is_set = True
        item.size = (1, 1, 1)  # TODO: Check new size against the item.
        dtype = (np.float32 if item.data_type == 'float' else np.uint64)
        self.__write_data_set(name, item, np.array(value, dtype=dtype).reshape(item.size))

    def __write_field(self, name, data):
        if name not in self.file_state:
            return

        item = self.file_state[name]
        item.is_set = True
        item.size = KWaveInputFile.__make_nd_tuple(data.shape, 3)  # TODO: Check new size against the item.
        dtype = (np.float32 if item.data_type == 'float' else np.uint64)
        self.__write_data_set(name, item, np.array(data, dtype=dtype).reshape(item.size))

    def __write_global_field(self, name, data):
        if not KWaveInputFile.__is_shape_compatible(self.domain_shape, data.shape):
            raise ValueError("Global field must have same size as the simulation domain!")

        self.__write_field(name, data)

    def __write_data_set(self, data_set_name, data_set_item, data):
        if data.shape != data_set_item.size:
            print(data.shape)
            raise ValueError("Metadata and data shape mismatch!")

        data_t = self.__array_to_output_order(data)

        if data_set_name in self.file_handle:
            data_set = self.file_handle[data_set_name]
            if data_set.shape != data_t.shape:
                raise ValueError("Dataset cannot change shape once created!")
        else:
            data_set = self.file_handle.create_dataset(data_set_name, data_t.shape,
                                                       dtype=(np.float32 if data_set_item.data_type == 'float' else np.uint64))

        data_set[:] = data_t

        KWaveInputFile.__create_string_attrib(data_set, 'data_type', data_set_item.data_type)
        KWaveInputFile.__create_string_attrib(data_set, 'domain_type', data_set_item.domain_type)

    def __initialize_file_state(self):
        self.file_state = {}

    def __update_file_state(self, modified_state):
        self.file_state.update({k: v for k, v in modified_state.items() if k not in self.file_state})

    def __write_grid_properties(self):
        self.__update_file_state(self.data_set_db.get_grid_properties())

        domain_shape = self.__shape_to_output_order(self.domain_shape)
        self.__write_scalar('Nx', domain_shape[0])
        self.__write_scalar('Ny', domain_shape[1])
        self.__write_scalar('Nz', domain_shape[2])

        spatial_delta = self.__shape_to_output_order(self.spatial_delta)
        self.__write_scalar('dx', spatial_delta[0])
        self.__write_scalar('dy', spatial_delta[1])
        self.__write_scalar('dz', spatial_delta[2])

        self.__write_scalar('Nt', self.nt)
        self.__write_scalar('dt', self.time_delta)

    def __write_pml_properties(self):
        self.__update_file_state(self.data_set_db.get_pml_variables())

        pml_size = self.__shape_to_output_order(self.pml_size)
        self.__write_scalar('pml_x_size', pml_size[0])
        self.__write_scalar('pml_y_size', pml_size[1])
        self.__write_scalar('pml_z_size', pml_size[2])

        pml_alpha = self.__shape_to_output_order(self.pml_alpha)
        self.__write_scalar('pml_x_alpha', pml_alpha[0])
        self.__write_scalar('pml_y_alpha', pml_alpha[1])
        self.__write_scalar('pml_z_alpha', pml_alpha[2])

    def __write_file_headers(self):
        headers = self.data_set_db.get_file_headers()

        for name, value in headers.items():
            self.__create_string_attrib(self.file_handle, name, value)

    def __finalize_and_validate_file(self):
        self.__update_file_state(self.data_set_db.get_simulation_flags())

        self.__write_file_headers()
        self.__write_grid_properties()
        self.__write_pml_properties()

        def is_set_safe(name, state):
            return name in state and state[name].is_set

        def get_shape_if_set(name, state, default=(0, 0, 0)):
            return state[name].size if (name in state and state[name].size is not None) else default

        self.__write_scalar('ux_source_flag',         1 if is_set_safe('ux_source_input', self.file_state) else 0)
        self.__write_scalar('uy_source_flag',         1 if is_set_safe('uy_source_input', self.file_state) else 0)
        self.__write_scalar('uz_source_flag',         1 if is_set_safe('uz_source_input', self.file_state) else 0)
        self.__write_scalar('p_source_flag',          get_shape_if_set('p_source_input',  self.file_state)[1])
        self.__write_scalar('p0_source_flag',         1 if is_set_safe('p0_source_input', self.file_state) else 0)
        self.__write_scalar('transducer_source_flag', 1 if is_set_safe('transducer_source_input', self.file_state) else 0)
        self.__write_scalar('nonuniform_grid_flag',   0)
        self.__write_scalar('nonlinear_flag',         1 if is_set_safe('BonA', self.file_state) else 0)
        self.__write_scalar('absorbing_flag',         1 if is_set_safe('alpha_coef',      self.file_state) else 0)
        self.__write_scalar('axisymmetric_flag',      0)

        # for k, v in self.file_state.items():
        #     print("{0}: {1}".format(k, str(v)))

    @staticmethod
    def __create_string_attrib(data_set, attr_name, string):
        np_string = np.string_(string)
        tid = h5py.h5t.C_S1.copy()
        tid.set_size(len(string) + 1)
        data_set.attrs.create(attr_name, np_string, dtype=h5py.Datatype(tid))

    @staticmethod
    def __make_nd_tuple(value, n, fill=1):
        if isinstance(value, tuple):
            return value + tuple([fill for _ in range(len(value), n)])

        return tuple([value for _ in range(0, n)])

    @staticmethod
    def __is_shape_compatible(shape_a, shape_b):
        max_len = max(len(shape_a), len(shape_b))
        shape_a = shape_a + tuple([1 for _ in range(len(shape_a), max_len)])
        shape_b = shape_b + tuple([1 for _ in range(len(shape_b), max_len)])

        return shape_a == shape_b
