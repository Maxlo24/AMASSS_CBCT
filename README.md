# Automatic Multi-Anatomical Skull Structure Segmentation of Cone-Beam Computed Tomography scans Using 3D UNETR
*AMASSS-CBCT*

Authors:
<font size="1"> Maxime Gillot (University of Michigan), Baptiste Baquero (UoM), Celia Le (UoM), Romain Deleat-Besson (UoM), Lucia Cevidanes (UoM), Jonas Bianchi (UoM), Marcela Gurgel (UoM), Marilia Yatabe (UoM), Najla Al Turkestani (UoM), Kayvan Najarian (UoM), Reza Soroushmehr (UoM), Steve Pieper (ISOMICS), Ron Kikinis (Harvard Medical School), Beatriz Paniagua ( Kitware ), Jonathan Gryak (UoM), Marcos Ioshida (UoM), Camila Massaro (UoM), Liliane Gomes (UoM), Heesoo Oh (University of Pacific), Karine Evangelista (UoM), Cauby Chaves Jr (University of Ceara), Daniela Garib (University of São Paulo), Fábio Costa (University of Ceara), Erika Benavides (UoM), Fabiana Soki (UoM), Jean-Christophe Fillion-Robin (Kitware), Hina Joshi (University of North Narolina), Juan Prieto (Dept. of Psychiatry UNC at Chapel Hill) </font> 


- The segmentation of medical and dental images is a fundamental step in automated clinical decision support systems. It supports the entire clinical workflows from diagnosis, therapy planning, intervention and follow-up. 
- We propose a novel tool to accurately process a full face segmentation in about 5 minutes that would otherwise require an average of 7h of manual work by experienced clinicians. 
- This work focuses on the integration of the state-of-the-art UNEt TRansformers (UNETR) of the Medical Open Network for Artificial Intelligence (MONAI) framework. 
- We trained and tested our models using 618 de-identified Cone-Beam Computed Tomography (CBCT) volumetric images of the head acquired with several parameters from different centers for a generalized clinical application. 
- Our results on a 5 fold cross-validation showed high accuracy and robustness with an F1 score up to $0.962\pm0.02$.
- Full face model made by combining the mandible, the maxilla, the cranial base, the cervical vertebra and the skin segmentation:

![Segmentation](https://user-images.githubusercontent.com/46842010/155926868-ca81d82b-8735-4f33-97af-0c3d616e6910.png)


# Running the code

## Using Docker
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

**Segmentation options/arguments**
- By default, the mandible (MAND), the maxilla (MAX), the cranial base (CB), the cervical vertebra (CV) and the skin (SKIN) structures  are segmented and a merged segmentation is generated.
    To choose which structure to segment, you can use the following arguments:
    ```
    -ss MAND MAX CB CV SKIN
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

## On your computer
## Prerequisites

python 3.8.8 with the librairies:

**Main librairies:**

> monai==0.7.0 \
> torch==1.10.1 \
> itk==5.2.1 \
> numpy==1.20.1 \
> simpleitk==2.1.1

**Pre-process**

To run the preprocess to organise the files and set them at the wanted spacing:

```
python3 init_training_data.py -i "path of the input folder with the scans and the segs" -o "path of the output folder"
```
By defaul the spacing is set at 0.5 but we can change and add other spacing with the argument :
```
-sp 0.X1 0.X2 ... 0.Xn 
````

To run the preprocess to correct the image contrast and fill the holes in the segmentations , run the folowing command line:

```
python3 correct_file.py -i "path of the input folder with the scans and the segs" -o "path of the output folder"
```
Expected results of the contrast adjustment :
![ContrastAdjust](https://user-images.githubusercontent.com/46842010/155178176-7e735867-4ad2-412d-9ac0-c47fe9d7cd8e.png)


___


Prediction steps

![prediction](https://user-images.githubusercontent.com/46842010/155927157-19206e54-7a90-4816-8eb7-72369a04c39e.png)

Results 

![RESULTS](https://user-images.githubusercontent.com/46842010/155927668-906b4fae-4249-4556-a4fa-7a622e9c6c81.png)


