# Automatic Multi-Anatomical Skull Structure Segmentation of Cone-Beam Computed Tomography scans Using 3D UNETR
*AMASSS-CBCT*


# Presentation

- The segmentation of medical and dental images is a fundamental step in automated clinical decision support systems. It supports the entire clinical workflows from diagnosis, therapy planning, intervention and follow-up. 
- We propose a novel tool to accurately process a full face segmentation in about 5 minutes that would otherwise require an average of 7h of manual work by experienced clinicians. 
- This work focuses on the integration of the state-of-the-art UNEt TRansformers (UNETR) of the Medical Open Network for Artificial Intelligence (MONAI) framework. 
- We trained and tested our models using 618 de-identified Cone-Beam Computed Tomography (CBCT) volumetric images of the head acquired with several parameters from different centers for a generalized clinical application. 
- Our results on a 5 fold cross-validation showed high accuracy and robustness with an F1 score up to $0.962\pm0.02$.
- Full face model made by combining the mandible, the maxilla, the cranial base, the cervical vertebra and the skin segmentation:

---
![Segmentation](https://user-images.githubusercontent.com/46842010/155926868-ca81d82b-8735-4f33-97af-0c3d616e6910.png)


# How to use AMASSS-CBCT

- Using the [AMASSS-CBCT 3D Slicer module](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools). You can segment a CBCT scan with no coding language required using 3D Slicer GUI.
- Using the [DSCI](https://dsci.dent.umich.edu/#/) web-based plateform, you can segment a CBCT scan with no coding language required.
- Localy, you can use the scipts on you computer using docker or the source code on github.


# Local usage

## Arguments to run AMASSS-CBCT prediction script

Recquired arguments:
```
-i <Path of the scan to segment> or <Folder with the scans to segment> 
-o <Output folder to save the segmentation>
-dm <Folder with the trained models>
```

Segmentation options:
```
 -ss <structure to segment>
 -hd <working on small FOV ?> 
 -m <merge the segmentation ?>
```

Valid arguments for `-ss`: 
- `MAND` (mandible)
- `MAX` (maxilla)
- `CB` (cranial base)
- `CV` (cervical vertebra)
- `UAW` (upper airway) 
- `SKIN` (skin)
- `RC` (root canal)



Save options:
```
-sf <Save outputs in one folder ?>
-id <ID in the generated files name>
```

3D surface generation arguments:
```
-vtk <Generate vtk files ?>
-vtks <Smoothing applied to the 3d surface>
```

Technical arguments:
```
-sp <Spacing recquired by the model>
-cs <Crop size as imput of the model>
-pr <Overlapping of the sliding window inferer (between 0 and 1)>
-mo <Merging order of the sgemented structure>
```

Computing power arguments:

```
-ncw <number of CPU processes>
-ngw <number of GPU processes> 
```


## Use Docker image
You can get the AMASSS docker image by running the folowing command lines.

**Building using the DockerFile**

From the DockerFile directory:
```
docker pull dcbia/amasss:latest
```

From the DockerFile directory:

```
docker build -t amasss .
```

**Automatic segmentation**
*Running on CPU*
```
docker run --rm --shm-size=5gb -v <Folder_with_the_scans_to_segment>:/app/data/scans amasss:latest python3 /app/MULTI_SEG/src/predict_CBCTSeg.py 
```
*Running on GPU*
```
docker run --rm --shm-size=5gb --gpus all -v <Folder_with_the_scans_to_segment>:/app/data/scans amasss:latest python3 /app/MULTI_SEG/src/predict_CBCTSeg.py
```

**Informations**
- A ***test scan*** "MG_scan_test.nii.gz" is provided in the Data folder of the AMASSS repositorie.
- If the prediction with the ***GPU is not working***, make sure you installed the NVIDIA Container Toolkit : 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

**Segmentation options/arguments exemple**
- By default, the mandible (MAND), the maxilla (MAX), the cranial base (CB), the cervical vertebra (CV) and the upper airway (UAW) structures  are segmented and a merged segmentation is generated.
    To choose which structure to segment, you can use the following arguments:
    ```
    -ss MAND MAX CB CV UAW
    ```
    To deactivate the merging step, you can use the following argument:
    ```
    -m False
    ```
- By default the prediction will use 5 CPU process and use a batch size of 5 on the GPU (which requires around 8GB on the GPU), you can use the following argument to change this numbers: 
    ```
    -ncw 2 -ngw 2
    ```
    (ncw for the CPU and ngw for the GPU)

___

## Using the scripts
## Prerequisites

python 3.8.8 with the librairies:

**Main librairies:**

> monai==0.7.0 \
> torch==1.10.1 \
> itk==5.2.1 \
> numpy==1.20.1 \
> simpleitk==2.1.1

You can install the required librairies by running the following command:

```
pip install -r requirements.txt
``` 


## Running the code

Using the script [predict_CBCTSeg.py](MULTI_SEG/src/predict_CBCTSeg.py)

Basic usage:
```
python3 predict_CBCTSeg.py -i <Folder_with_the_scans_to_segment> -o <Folder_to_save_the_segmentation> -dm <Folder with the trained models>
```
Example to segment the mandible, the maxilla, the cranial base, the cervical vertebra and the upper airway in a large FOV scan (the MG_scan_test.nii.gz) with 3D surface generation and merging:

```
python3 predict_CBCTSeg.py -i ./Data/MG_scan_test.nii.gz -o ./Data -dm <Folder with the trained models> -ss MAND MAX CB CV UAW -vtk True
```

Prediction steps

![prediction](https://user-images.githubusercontent.com/46842010/155927157-19206e54-7a90-4816-8eb7-72369a04c39e.png)



# Train AMASSS-CBCT

### Prepare the data

**Spacing**
To run the preprocess to organise the files and set them at the wanted spacing:

Change the scan spacing to the wanted spacing using the script [init_training_data.py](MULTI_SEG/src/init_training_data.py):

```
python3 init_training_data.py -i "path of the input folder with the scans and the segs" -o "path of the output folder"
```
By defaul the spacing is set at 0.5 but we can change and add other spacing with the argument :
```
-sp 0.X1 0.X2 ... 0.Xn 
````

**Contrast adjustment**
To run the preprocess to correct the image contrast and fill the holes in the segmentations
run the folowing command line using the script [correct_file.py](MULTI_SEG/src/correct_file.py):

```
python3 correct_file.py -i <Path of the input folder with the scans and the segs> -o <Path of the output folder> -rad <Closing radius>
```

Expected results of the contrast adjustment :
![ContrastAdjust](https://user-images.githubusercontent.com/46842010/155178176-7e735867-4ad2-412d-9ac0-c47fe9d7cd8e.png)


### Organise the training folder

Organise the training folder as follows:

<img width="278" alt="Screen Shot 2022-02-22 at 12 11 56 PM" src="https://user-images.githubusercontent.com/46842010/181832258-ae140fed-4e8a-4871-a3b9-6e7dd68f628c.png">


### Start the training

Use the script [train_CBCTseg.py](MULTI_SEG/src/train_CBCTseg.py):


```
python3 train_CBCTseg.py --dir_project <Path of the training folder>

```
Aditional options:
```
 -mn <Model name>
-vp <Porcentage of data for validation> 
-cs <Input size of the model> 
-me <number of epoch> 
-nl <number of output label> 
-bs <batch size> 
-nw <number of workers>
```

You can launch a TensorBoard session to follow the training progress:


___



Results 

![RESULTS](https://user-images.githubusercontent.com/46842010/155927668-906b4fae-4249-4556-a4fa-7a622e9c6c81.png)




# Acknowledgements

Authors: Maxime Gillot (University of Michigan), Baptiste Baquero (UoM), Celia Le (UoM), Romain Deleat-Besson (UoM), Lucia Cevidanes (UoM), Jonas Bianchi (UoM), Marcela Gurgel (UoM), Marilia Yatabe (UoM), Najla Al Turkestani (UoM), Kayvan Najarian (UoM), Reza Soroushmehr (UoM), Steve Pieper (ISOMICS), Ron Kikinis (Harvard Medical School), Beatriz Paniagua ( Kitware ), Jonathan Gryak (UoM), Marcos Ioshida (UoM), Camila Massaro (UoM), Liliane Gomes (UoM), Heesoo Oh (University of Pacific), Karine Evangelista (UoM), Cauby Chaves Jr (University of Ceara), Daniela Garib (University of São Paulo), Fábio Costa (University of Ceara), Erika Benavides (UoM), Fabiana Soki (UoM), Jean-Christophe Fillion-Robin (Kitware), Hina Joshi (University of North Narolina), Juan Prieto (Dept. of Psychiatry UNC at Chapel Hill)

Supported by NIDCR R01 024450, AA0F Grabber Family Teaching and Research Award and by Research Enhancement Award Activity 141 from the University of the Pacific, Arthur A. Dugoni School of Dentistry.
