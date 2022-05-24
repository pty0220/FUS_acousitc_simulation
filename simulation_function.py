import os
import numpy as np
import math
import time
import SimpleITK as sitk

from help_function.niiCook import niiCook
from help_function import help_function as hlp

from kwave_function.kwave_input_file import KWaveInputFile
from kwave_function.kwave_output_file import KWaveOutputFile, DomainSamplingType, SensorSamplingType
from kwave_function.kwave_bin_driver import KWaveBinaryDriver

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

start = time.time()
current_path = os.path.dirname(__file__)

class makeSimulation():

    def __init__(self, path=False):

        print("Check init")
        ####################################################################
        # Material properties
        self.c_water = 1482  # [m/s]
        self.d_water = 1000  # [kg/m^3]
        self.a_water = 0.0253  # [Np/MHz/cm]

        self.c_bone = 3100  # [m/s]    # 2800 or 3100 m/s
        self.d_bone = 2200  # [kg/m^3]
        self.a_bone_min = 21.5  # [Np/MHz/cm]
        self.a_bone_max = 208.9  # [Np/MHz/cm]

        self.alpha_power = 1.01


        ####################################################################
        # Source properties
        self.amplitude = 1  # source pressure [Pa]
        self.source_freq = 25e4  # frequency [Hz]
        self.ROC = 99  # [mm]     # transducer setting
        self.width = 95  # [mm]
        self.focal_length = 85

        ####################################################################
        # Bounary condition
        self.boundary = 0

        ####################################################################
        # Time step
        self.CFL = 0.1
        self.end_time = 100e-6
        self.points_per_wavelength = np.pi*2 # number of grid points per wavelength at f0

        ####################################################################
        # Recording
        self.recording = False

        ####################################################################
        # Back propagation
        self.PHASE = []
        self.AMP = []
        self.back_source = []
        self.optimizer_check = 0

        ####################################################################
        # Path
        if path == False:
            self.path = current_path
        else:
            self.path = path
            
            try:
                os.mkdir(self.path)
                print("Make save Path: ", self.path)

            except:
                print("Path: ", self.path, "/ already exist")

    ################################################################################################################################################
    # Forward simulation
    ################################################################################################################################################

    # Make simulation input -> Crop and Resample
    def preprocessing(self, itk_image, target_pose):

        target_pose = np.multiply(target_pose, (-1, -1, 1)).astype(float)

        ####################################################################
        # Source properties
        dx = self.c_water / (self.points_per_wavelength * self.source_freq)  # [m]
        dy = dx
        dz = dx

        ####################################################################
        # Grid_size contains the PML (default 20)
        grid_res = (dx, dy, dz)

        ####################################################################
        # Skull process
        simul_domain = niiCook()
        try:
            simul_domain.readSavedFile(itk_image)
        except:
            simul_domain.readITK(itk_image)
        skullCrop_itk, skullCrop_arr = simul_domain.makeSimulationDomain(grid_res, self.focal_length, target_pose)
        del simul_domain

        print("Perform skull processing")
        print("Simulation domain: ", skullCrop_arr.shape)
        print("Simulation dx: ", dx)

        self.domainCook = niiCook()
        self.domainCook.readITK(skullCrop_itk)
        self.domainCook.saveITK(self.path+"\\skullCrop_itk.nii")

        self.dx = dx
        self.grid_res = grid_res
        self.target_pose = target_pose
        self.skullCrop_arr = skullCrop_arr
        self.rawCrop_arr = skullCrop_arr.copy()
        self.skullCrop_itk = skullCrop_itk
        self.domain_shape = skullCrop_arr.shape
        self.target_idx = np.array(skullCrop_itk.TransformPhysicalPointToIndex(target_pose)).astype(int)
        self.p0 = np.zeros(self.skullCrop_arr.shape)

    def preprocessing_rotate(self, itk_image, tran_pose, target_pose):
        s = time.time()

        tran_pose = np.multiply(tran_pose, (-1, -1, 1)).astype(float)
        target_pose = np.multiply(target_pose, (-1, -1, 1)).astype(float)

        ####################################################################
        ####################################################################
        # Source properties
        dx = self.c_water / (self.points_per_wavelength * self.source_freq)  # [m]
        dy = dx
        dz = dx

        ####################################################################
        # Grid_size contains the PML (default 20)
        grid_res = (dx, dy, dz)

        ####################################################################
        # Skull process

        simul_domain = niiCook()
        if isinstance(itk_image, str):
            simul_domain.readSavedFile(itk_image)
        else:
            simul_domain.readITK(itk_image)

        skullCrop_arr, skullCrop_itk = simul_domain.makeSimulationDomain_rotate(grid_res, tran_pose, target_pose, self.focal_length, self.width)

        self.domainCook = niiCook()
        self.domainCook.readITK(skullCrop_itk)
        self.domainCook.saveITK(self.path+"\\skullCrop_rotate_itk.nii")

        e = time.time()

        print("Perform skull processing")
        print("Simulation domain: ", skullCrop_arr.shape)
        print("Simulation dx: ", dx)
        print("Resample time:",e-s)

        self.dx = dx
        self.grid_res = grid_res
        self.target_pose = target_pose
        self.skullCrop_arr = skullCrop_arr
        self.rawCrop_arr = skullCrop_arr.copy()
        self.skullCrop_itk = skullCrop_itk
        self.domain_shape = skullCrop_arr.shape
        self.target_idx = np.array(skullCrop_itk.TransformPhysicalPointToIndex(target_pose)).astype(int)
        self.p0 = np.zeros(self.skullCrop_arr.shape)

    # Read pre-processed image
    def read_preprocessing(self, itk_image):

        domainCook = niiCook()
        try:
            domainCook.readSavedFile(itk_image)
        except:
            domainCook.readITK(itk_image)

        domainCook.makeITK(domainCook.array, self.path+"\\skullCrop_itk.nii")

        dx = domainCook.spacing[0]/1000
        dy = dx
        dz = dx
        grid_res = (dx, dy, dz)

        self.dx = dx
        self.grid_res = grid_res
        self.target_idx = l2n(domainCook.dimension)/2
        self.target_idx = (int(self.target_idx[0]), int(self.target_idx[1]), int(self.target_idx[2]))
        self.target_pose = np.array(domainCook.itkImage.TransformIndexToPhysicalPoint(self.target_idx))
        self.skullCrop_arr = domainCook.array
        self.rawCrop_arr = domainCook.array.copy()
        self.skullCrop_itk = domainCook.itkImage
        self.domain_shape = domainCook.dimension
        self.domainCook = domainCook
        self.p0 = np.zeros(self.skullCrop_arr.shape)

    # Make transducer at given position
    def make_transducer(self, tran_pose, normal = l2n([0,0,0])):

        width = self.width
        ROC = self.ROC
        dx = self.dx

        ## Slicer to NIFTI coordinate
        tran_pose = np.multiply(tran_pose,  (-1, -1, 1)).astype(float)
        self.tran_pose = tran_pose

        self.tran_idx = np.array(self.skullCrop_itk.TransformPhysicalPointToIndex(tran_pose)).astype(int)

        if np.all(normal ==0):
            self.normal = (self.target_idx - self.tran_idx)/np.linalg.norm(self.target_idx - self.tran_idx)
        else:
            self.normal =l2n(normal)

        Tcenter = self.tran_idx
        Tnormal =  self.normal

        Spos = hlp.make_transducer(ROC, width, dx, Tcenter, Tnormal)
        Spos = Spos.astype(int)

        if np.any(Spos[:,0] >= self.skullCrop_arr.shape[0])\
                or np.any(Spos[:,1] >= self.skullCrop_arr.shape[1])\
                or np.any(Spos[:,2] >= self.skullCrop_arr.shape[2]):
            self.Spos = -10
            self.p0 = np.ones(self.domain_shape)*(-10)
        else:
            p0 = self.skullCrop_arr.copy()
            p0[:,:,:] = 0
            p0[Spos[:,0],Spos[:,1],Spos[:,2]] = 1

            self.Spos = Spos
            self.p0 = p0

            self.trans_itk = self.domainCook.makeITK(self.p0*2000, self.path+"\\transducer.nii")

    # Run simulation
    def run_simulation(self):
        start = time.time()
        print(" ")
        print(" ")
        print("################################")
        print("Start simulation")
        print("################################")
        print(" ")
        print(" ")
        print("####  Simulation specs  ####")
        print("Iso Voxel size: " + str(self.dx))

        print("CFL: " + str(self.CFL))
        print("end time: " + str(self.end_time))
        print("PPW: " + str(self.points_per_wavelength))

        ####################################################################
        # Source properties
        amplitude = self.amplitude       # source pressure [Pa]
        source_freq = self.source_freq     # frequency [Hz]

        ####################################################################
        # Material properties
        c_water = self.c_water      # [m/s]
        d_water = self.d_water      # [kg/m^3]
        a_water = self.a_water   # [Np/MHz/cm]

        c_bone = self.c_bone       # [m/s]    # 2800 or 3100 m/s
        d_bone = self.d_bone       # [kg/m^3]
        a_bone_min = self.a_bone_min   # [Np/MHz/cm]
        a_bone_max = self.a_bone_max  # [Np/MHz/cm]
        alpha_power = self.alpha_power

        ####################################################################
        # Grid properties
        grid_res = self.grid_res

        ####################################################################
        # skull array
        skullCrop_arr = self.skullCrop_arr

        ####################################################################
        # Transducer
        p0 = self.p0


        ####################################################################
        # Time step
        CFL      = self.CFL
        end_time = self.end_time
        dt       = CFL * grid_res[0] / c_water
        steps    = int(end_time / dt)


        input_filename  ='kwave_in.h5'
        output_filename ='kwave_out.h5'

        ####################################################################
        # Skull process
        grid_size = skullCrop_arr.shape
        skullCrop_arr[skullCrop_arr < 250] = 0

        ####################################################################
        # assign skull properties depend on HU value  - Ref. Numerical evaluation, Muler et al, 2017
        if np.all(skullCrop_arr==0):
            skull_max = 1  # free water case
        else:
            skull_max = np.max(skullCrop_arr)

        input_X = (skullCrop_arr/skull_max) - p0 # Make simulation input X
        self.domainCook.makeITK(input_X, os.path.join(self.path, "input_X.nii"))

        input_X[input_X<0] = 0 # Extract transducer

        PI = 1 - input_X
        ct_sound_speed = c_water*PI + c_bone*(1-PI)
        ct_density  = d_water*PI + d_bone*(1-PI)
        ct_att          = a_bone_min + (a_bone_max-a_bone_min)*np.power(PI, 0.5)
        ct_att[PI==1]   = a_water

        ###################################################################
        # assign skull properties depend on HU value  - Ref. Multi resolution, Yoon et al, 2019
        # PI = skullCrop_arr/np.max(skullCrop_arr)
        # ct_sound_speed = c_water + (2800 - c_water)*PI
        # ct_density     = d_water + (d_bone - d_water)*PI
        # ct_att         = 0 + (20 - 0)*PI

        ####################################################################
        # Assign material properties
        sound_speed     = ct_sound_speed
        density         = ct_density
        alpha_coeff_np  = ct_att

        alpha_coeff = hlp.neper2db(alpha_coeff_np*source_freq/1e6/pow(2*np.pi*source_freq, alpha_power), alpha_power)

        ####################################################################
        # Define simulation input and output files
        print("## k-wave core input function")
        input_file  = KWaveInputFile(input_filename, grid_size, steps, grid_res, dt)
        output_file = KWaveOutputFile(file_name=output_filename)


        ####################################################################
        # Transducer signal
        source_signal = amplitude * np.sin((2*math.pi)*source_freq*np.arange(0.0, steps*dt, dt))
        self.source_signal = source_signal
        result_name = "forward.nii"

        ####################################################################
        # Open the simulation input file and fill it as usual
        with input_file as file:
            file.write_medium_sound_speed(sound_speed)
            file.write_medium_density(density)
            file.write_medium_absorbing(alpha_coeff, alpha_power)
            file.write_source_input_p(file.domain_mask_to_index(p0), self.source_signal, KWaveInputFile.SourceMode.ADDITIVE, c_water)
            sensor_mask = np.ones(grid_size)
            file.write_sensor_mask_index(file.domain_mask_to_index(sensor_mask))

        # Create k-Wave solver driver, which will call C++/CUDA k-Wave binary.
        # It is usually necessary to specify path to the binary: "binary_path=..."
        driver = KWaveBinaryDriver()

        # Specify which data should be sampled during the simulation (final pressure in the domain and
        # RAW pressure at the sensor mask
        driver.store_pressure_everywhere([DomainSamplingType.MAX])
        if self.recording:
            driver.store_pressure_at_sensor([SensorSamplingType.RAW])

        # Execute the solver with specified input and output files
        driver.run(input_file, output_file)
        print("## Calculation time :", time.time() - start)

        #Open the output file and generate plots from the results
        with output_file as file:
            if self.recording:
                p_raw_raw = file.read_pressure_at_sensor(SensorSamplingType.RAW)
                p_raw = np.squeeze(p_raw_raw).transpose([1,0])
                del p_raw_raw

                time_step = p_raw.shape[1]
                p_raw = np.ravel(p_raw)
                p_raw = np.reshape(p_raw, (self.domain_shape[2], self.domain_shape[1], self.domain_shape[0], time_step))
                p_raw = p_raw.transpose([2, 1, 0, 3])
                self.p_raw = p_raw
                self.p_max = np.max(p_raw, axis=3)
                np.save(os.path.join(self.path, "forward_recording"), self.p_raw)

            else:
                self.p_max = file.read_pressure_everywhere(DomainSamplingType.MAX)

            result_itk = self.domainCook.makeITK(self.p_max, os.path.join(self.path, result_name))
            self.result_itk = result_itk

        return result_itk

    # Run simulation in Free water
    def free_water_run_simulation(self):

        # random vector for vertical direction
        target_pose = [0,0,10]
        ####################################################################
        # Source properties
        dx = self.c_water / (self.points_per_wavelength * self.source_freq)  # [m]
        dy = dx
        dz = dx

        ####################################################################
        # Grid_size contains the PML (default 20)
        grid_res = (dx, dy, dz)
        simul_spacing = l2n(grid_res) * 1000

        ####################################################################
        # Make reference image
        Nx = np.ceil(self.width/simul_spacing[0]) + 10 #PML
        Ny = Nx
        Nz = np.ceil((self.focal_length*1.7)/simul_spacing[2]) + 20 #PML

        domain = l2n([Nx, Ny, Nz])
        domain = domain - domain % 10

        x_end = -simul_spacing[0] * domain[0] / 2
        y_end = -simul_spacing[1] * domain[1] / 2
        z_end = -simul_spacing[2] * 20

        grid_origin = (x_end, y_end, z_end)

        reference_image = sitk.Image(int(domain[0]), int(domain[1]), int(domain[2]), sitk.sitkFloat32)
        reference_image.SetSpacing(simul_spacing)
        reference_image.SetOrigin(grid_origin)
        reference_image[:, :, :] = 0

        skullCrop_itk = reference_image
        skullCrop_arr = sitk.GetArrayFromImage(skullCrop_itk)

        ####################################################################
        # Save
        self.domainCook = niiCook()
        self.domainCook.readITK(skullCrop_itk)
        self.domainCook.saveITK(self.path+"\\skullCrop_rotate_itk.nii")

        self.dx = dx
        self.grid_res = grid_res
        self.target_pose = target_pose
        self.skullCrop_arr = skullCrop_arr
        self.rawCrop_arr = skullCrop_arr.copy()
        self.skullCrop_itk = skullCrop_itk
        self.domain_shape = skullCrop_arr.shape
        self.target_idx = np.array(skullCrop_itk.TransformPhysicalPointToIndex(target_pose)).astype(int)
        self.p0 = np.zeros(self.skullCrop_arr.shape)

        ####################################################################
        # Run simulation from free water // Transducer position is located at [0,0,0]
        self.make_transducer([0,0,0])
        self.run_simulation()
