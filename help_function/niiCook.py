
import vtk
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage
from vtk.util import numpy_support as ns

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

class niiCook():

    def __init__(self):
        self.dummy = 0

    def readITK(self, image):

        spacing   = image.GetSpacing()
        origin    = image.GetOrigin()  ## bounds
        dimension = image.GetSize()
        direction = image.GetDirection()
        extent = (0, dimension[0] - 1, 0, dimension[1] - 1, 0, dimension[2] - 1)

        array = sitk.GetArrayFromImage(image)

        self.itkImage = image
        self.spacing = spacing
        self.origin = origin
        self.dimension = dimension
        self.extent = extent
        self.array = array
        self.direction = direction

    def readSavedFile(self, filePath):

        if filePath[-2:] == "gz":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(filePath)

        elif filePath[-3:] == "nii":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(filePath)

        elif filePath[-4:] == "nrrd":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NrrdImageIO")
            reader.SetFileName(filePath)

        else:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filePath)
            reader.SetFileNames(dicom_names)

        image     = reader.Execute()
        spacing   = image.GetSpacing()
        origin    = image.GetOrigin()  ## bounds
        dimension = image.GetSize()
        direction = image.GetDirection()
        extent = (0, dimension[0] - 1, 0, dimension[1] - 1, 0, dimension[2] - 1)

        array = sitk.GetArrayFromImage(image)

        self.direction = direction
        self.itkImage = image
        self.spacing = spacing
        self.origin = origin
        self.dimension = dimension
        self.extent = extent
        self.array = array

    def cropVolume(self, dimension, origin, spacing, original_image = 0):

        reference_image = sitk.Image(int(dimension[0]), int(dimension[1]), int(dimension[2]), sitk.sitkFloat32)
        reference_image.SetSpacing(spacing)
        reference_image.SetOrigin(origin)
        reference_image[:,:,:] = 0

        if original_image == 0:
            original_image = self.itkImage

        rigid_euler = sitk.Euler3DTransform()
        interpolator = sitk.sitkCosineWindowedSinc
        default_value = -1000.0

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(rigid_euler)
        resampler.SetNumberOfThreads(15)
        crop_image = resampler.Execute(original_image)

        return reference_image, crop_image

    def makeSimulationDomain(self, simul_spacing, focal_length, target_pos, make_even = True):

        simul_spacing = l2n(simul_spacing)*1000

        # for Optimal position 1.2

        max_point = target_pos + (focal_length * 1.2)
        min_point = target_pos - (focal_length * 1.2)

        bound = np.abs(max_point - min_point)
        domain = np.round(bound / simul_spacing)
        if make_even:
            domain = domain - domain % 10

        reference_image, crop_image = self.cropVolume(domain, min_point, simul_spacing)
        crop_array = sitk.GetArrayFromImage(crop_image)

        return crop_image, crop_array

    def makeSimulationDomain_rotate(self, simul_spacing, tran_pos, target_pos, focal_length, width, make_even=True):

        simul_spacing = l2n(simul_spacing) * 1000

        ####################################################################
        # Make reference image
        Nx = np.ceil(width/simul_spacing[0]) + 10 #PML
        Ny = Nx
        Nz = np.ceil((focal_length*1.7)/simul_spacing[2]) + 20 #PML

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


        ####################################################################
        # Make Transform matrix
        dir = tran_pos - target_pos

        n_z = np.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
        if n_z == 0:
            transform_matrix = np.eye(3)
        else:
            rotate_z = l2n(((dir[0] / n_z, dir[1] / n_z, 0),
                            (-dir[1] / n_z, dir[0] / n_z, 0),
                            (0, 0, 1)))

            dir2 = np.dot(rotate_z, np.transpose(dir))
            n_y = np.sqrt(dir2[0] * dir2[0] + dir2[2] * dir2[2])
            rotate_y = l2n(((-dir2[2] / n_y, 0, dir2[0] / n_y),
                            (0, 1, 0),
                            (-dir2[0] / n_y, 0, -dir2[2] / n_y)))

            transform_matrix = np.dot(rotate_y, rotate_z)
            transform_matrix = np.linalg.inv(transform_matrix)


        move2tran = sitk.TranslationTransform(3, tran_pos)

        ####################################################################
        # Apply function
        rigid_euler = sitk.Euler3DTransform()
        rigid_euler.SetMatrix(transform_matrix.flatten())  # rotate rely on transducer axes axis
        rigid_euler.SetTranslation(move2tran.GetOffset())  # move to target
        interpolator = sitk.sitkCosineWindowedSinc

        # Resampling
        default_value = 0
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(rigid_euler)
        resampler.SetNumberOfThreads(10)
        if self.array.max()==1:
            crop_image = resampler.Execute(self.itkImage*2000)
            crop_image = crop_image/2000
        else:
            crop_image = resampler.Execute(self.itkImage)

        # Re-transform to origin image
        grid_origin_tran = np.dot(transform_matrix, grid_origin)
        crop_image.SetOrigin(grid_origin_tran+tran_pos)
        crop_image.SetDirection(n2l(transform_matrix.flatten()))

        # Get array
        crop_array = sitk.GetArrayFromImage(crop_image)

        return crop_array, crop_image

    def makeITK(self, array, path=False):

        result_itk = sitk.GetImageFromArray(array)
        result_itk.SetSpacing(self.spacing)
        result_itk.SetOrigin(self.origin)
        result_itk.SetDirection(self.direction)

        if path:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(path)
            writer.Execute(result_itk)
        return result_itk

    def saveITK(self, path=False):

        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.Execute(self.itkImage)
