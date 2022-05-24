# Acoustic simulation for transcranial focused ultrasound (tFUS) - Python

This package has the acoustic simulation for tFUS application with a single-element transducer.
The simulation uses the k-Wave Matlab toolbox to solve the linear wave equation. 

Although the k-Wave was written in Matlab and C++, in this package, 
we implemented the python wrapper and developed the several help functions for the tFUS application

## Structure
    help_function      --> Folder for help function  
    kwave_core         --> Folder for k-Wave core (e.g., .exe and .dll)
    kwave_function     --> Folder for python wapper (e.g., Matlab to Python) author by Filip Vaverka
    Test_data          --> Test skull data 
    
    simulation_function.py   --> main simulation function 
    
    example_free_water.py    --> simulation at free water 
    example_with_skull       --> simulation for tFUS application 

 ## Simulation process
 
 #### 1. Set transducer spec (e.g., ROC, width, FF, and focal length)
 
 #### 2. Set sonication spec  (e.g, end_time, CFL, and PPW etc)

 #### 3. Pre-process (Crop and Resample --> make simulation domain)
 + Using the CT image, the simulation domain with the skull was cropped and resampled.
 + The pre-process was performed using "Simpleitk" resampling strategy (e.g., nearest neighbor)
 
 #### 4. Make transducer 
 + On the simulation domain the transducer was made depending on the given position and orientation of the transducer.
 
 #### 5. Run simulation
 + Houns field Unit (HU) was converted to acoustic properties (speed of sound, density, and attenuation coefficient)
 + The simulation detail was shown in our paper -- Koh H, Park TY, Chung YA, Lee JH, Kim H. Acoustic Simulation for Transcranial Focused Ultrasound Using GAN-Based Synthetic CT. IEEE J Biomed Health Inform. 2022 Jan;26(1):161-171. doi: 10.1109/JBHI.2021.3103387. Epub 2022 Jan 17. PMID: 34388098. 

![Fig 1](https://user-images.githubusercontent.com/42193020/158503413-f517cd45-f192-497c-92f1-429ecb60df57.png)

 ## Data management
 
 #### + In this package, all 3D grid data (i.e., skull and simulation result) was saved as .nii file 
  + .nii file can save the 3D data with origin, spacing, and volume value (HU)
  + Easy to plot using medical image platform e.g., 3D Slicer
  
 #### + In result dir, saved file like this
    skullCrop_itk.nii   --> Pre-processed skull image (simulation domain)
    Transudcer.nii      --> Discritized transducer geometry at simulation domain
    forward.nii         --> forward acoustic simulation result 

 ## How to start
 + Create the environment using .yaml file. In ananconda prompte type this
  
        conda env create --file environment.yaml
        conda activate Simulation_env
 
 + Then, perform the "example_free_water.py" or "example_with_skull.py".

## Contact
This package was developed by TY Park from Korea Institute Science and Technology school (KIST school)
E-mail: pty0220@kist.re.kr
