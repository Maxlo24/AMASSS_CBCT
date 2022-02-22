# Multi-Anatomical Skull Structure Segmentation of Cone-Beam Computed Tomography scans Using 3D UNETR

Authors: Deleat-Besson Romain (UoM), Le Celia (UoM)

Scripts for Multi organ segmentation in CBCT

Full face model made by combining the mandible, the maxilla, the cranial base, the cervical vertebra and the skin segmentation using a MONAI UNETR :

![Segmentation](https://user-images.githubusercontent.com/46842010/155178127-61dfbf6b-9e4f-450b-966e-a2afca3fc4e4.png)

## Prerequisites

python 3.7.9 with the librairies:

**Main librairies:**

> tensorflow==2.4.1 \
> tensorflow-gpu==2.4.0 \
> Pillow==7.2.0 \
> numpy==1.19.5 \
> itk==5.2.0 

## Running the code

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
