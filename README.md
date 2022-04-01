# Automatic Multi-Anatomical Skull Structure Segmentation of Cone-Beam Computed Tomography scans Using 3D UNETR

Authors:
Maxime Gillot (University of Michigan), Baptiste Baquero (UoM), Celia Le (UoM), Romain Deleat-Besson (UoM), Lucia Cevidanes (UoM), Jonas Bianchi (UoM), Marcela Gurgel (UoM), Marilia Yatabe (UoM), Najla Al Turkestani (UoM), Kayvan Najarian (UoM), Reza Soroushmehr (UoM), Steve Pieper (ISOMICS), Ron Kikinis (Harvard Medical School), Beatriz Paniagua ( Kitware ), Jonathan Gryak (UoM), Marcos Ioshida (UoM), Camila Massaro (UoM), Liliane Gomes (UoM), Heesoo Oh (University of Pacific), Karine Evangelista (UoM), Cauby Chaves Jr (University of Ceara), Daniela Garib (University of São Paulo), Fábio Costa (University of Ceara), Erika Benavides (UoM), Fabiana Soki (UoM), Jean-Christophe Fillion-Robin (Kitware), Hina Joshi (University of North Narolina), Juan Prieto (Dept. of Psychiatry UNC at Chapel Hill)


Scripts for Multi organ segmentation in CBCT

Full face model made by combining the mandible, the maxilla, the cranial base, the cervical vertebra and the skin segmentation using the MONAI UNETR :

![Segmentation](https://user-images.githubusercontent.com/46842010/155926868-ca81d82b-8735-4f33-97af-0c3d616e6910.png)


## Prerequisites

python 3.8.8 with the librairies:

**Main librairies:**

> monai==0.7.0 \
> torch==1.10.1 \
> itk==5.2.1 \
> numpy==1.20.1 \
> simpleitk==2.1.1

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

## Use Docker
You can get the AMASSS docker image by running the folowing command line:

```
docker pull dcbia/
```



Prediction steps

![prediction](https://user-images.githubusercontent.com/46842010/155927157-19206e54-7a90-4816-8eb7-72369a04c39e.png)

Results 

![RESULTS](https://user-images.githubusercontent.com/46842010/155927668-906b4fae-4249-4556-a4fa-7a622e9c6c81.png)


