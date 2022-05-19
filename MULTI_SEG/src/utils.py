import numpy as np  
import SimpleITK as sitk
import itk


import os
import matplotlib.pyplot as plt
from scipy import stats

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

import datetime
import glob
import sys
import cc3d
import shutil


# ----- MONAI ------

from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AddChannel,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandShiftIntensityd,
    ScaleIntensityd,
    ScaleIntensity,
    Spacingd,
    Spacing,
    Rotate90d,
    RandRotate90d,
    ToTensord,
    ToTensor,
    SaveImaged,
    SaveImage,
    RandCropByLabelClassesd,
    Lambdad,
    CastToTyped,
    SpatialCrop,
    BorderPadd,
    RandAdjustContrastd,
    HistogramNormalized,
    NormalizeIntensityd,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_TYPE = torch.float32



######## ########     ###    ##    ##  ######  ########  #######  ########  ##     ##  ######  
   ##    ##     ##   ## ##   ###   ## ##    ## ##       ##     ## ##     ## ###   ### ##    ## 
   ##    ##     ##  ##   ##  ####  ## ##       ##       ##     ## ##     ## #### #### ##       
   ##    ########  ##     ## ## ## ##  ######  ######   ##     ## ########  ## ### ##  ######  
   ##    ##   ##   ######### ##  ####       ## ##       ##     ## ##   ##   ##     ##       ## 
   ##    ##    ##  ##     ## ##   ### ##    ## ##       ##     ## ##    ##  ##     ## ##    ## 
   ##    ##     ## ##     ## ##    ##  ######  ##        #######  ##     ## ##     ##  ######  


def CreateTrainTransform(CropSize = [64,64,64],padding=10,num_sample=10):
    train_transforms = Compose(
        [
        LoadImaged(keys=["scan", "seg"]),
        AddChanneld(keys=["scan", "seg"]),
        BorderPadd(keys=["scan", "seg"],spatial_border=padding),
        ScaleIntensityd(
            keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
        ),
        # CropForegroundd(keys=["scan", "seg"], source_key="scan"),
        RandCropByPosNegLabeld(
            keys=["scan", "seg"],
            label_key="seg",
            spatial_size=CropSize,
            pos=1,
            neg=1,
            num_samples=num_sample,
            image_key="scan",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["scan", "seg"],
            spatial_axis=[0],
            prob=0.20,
        ),
        RandFlipd(
            keys=["scan", "seg"],
            spatial_axis=[1],
            prob=0.20,
        ),
        RandFlipd(
            keys=["scan", "seg"],
            spatial_axis=[2],
            prob=0.20,
        ),
        RandRotate90d(
            keys=["scan", "seg"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["scan"],
            offsets=0.10,
            prob=0.50,
        ),
        RandAdjustContrastd(
            keys=["scan"],
            prob=0.8,
            gamma = (0.5,2)
        ),
        ToTensord(keys=["scan", "seg"]),
        ]
    )

    return train_transforms

def CreateValidationTransform():

    val_transforms = Compose(
        [
            LoadImaged(keys=["scan", "seg"]),
            AddChanneld(keys=["scan", "seg"]),
            ScaleIntensityd(
                keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
            ),
            # CropForegroundd(keys=["scan", "seg"], source_key="scan"),
            RandFlipd(
                keys=["scan", "seg"],
                spatial_axis=[0],
                prob=0.20,
            ),
            RandFlipd(
                keys=["scan", "seg"],
                spatial_axis=[1],
                prob=0.20,
            ),
            RandFlipd(
                keys=["scan", "seg"],
                spatial_axis=[2],
                prob=0.20,
            ),
            RandRotate90d(
                keys=["scan", "seg"],
                prob=0.10,
                max_k=3,
            ),
            RandAdjustContrastd(
                keys=["scan"],
                prob=0.8,
                gamma = (0.5,2)
            ),
            ToTensord(keys=["scan", "seg"]),
        ]
    )


    # val_transforms = Compose(
    #     [
    #         LoadImaged(keys=["scan", "seg"]),
    #         AddChanneld(keys=["scan", "seg"]),
    #         ScaleIntensityd(
    #             keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
    #         ),
    #         ToTensord(keys=["scan", "seg"]),
    #     ]
    # )

    return val_transforms

def CreatePredTransform(spacing):
    pred_transforms = Compose(
        [
            LoadImaged(keys=["scan"]),
            AddChanneld(keys=["scan"]),
            ScaleIntensityd(
                keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
            ),
            Spacingd(keys=["scan"],pixdim=spacing),
            ToTensord(keys=["scan"]),
        ]
    )
    return pred_transforms

def CreatePredictTransform(data,spacing):

    pre_transforms = Compose(
        [AddChannel(),ScaleIntensity(minv = 0.0, maxv = 1.0, factor = None),ToTensor()]
    )

    input_img = sitk.ReadImage(data) 
    img = input_img
    img = ItkToSitk(Rescale(data,[spacing,spacing,spacing]))
    img = sitk.GetArrayFromImage(img)
    # img = CorrectImgContrast(img,0.,0.99)
    pre_img = pre_transforms(img)
    pre_img = pre_img.type(DATA_TYPE)
    return pre_img,input_img


def CorrectImgContrast(img,min_porcent,max_porcent):
    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    return img

######## ########     ###    #### ##    ## #### ##    ##  ######   
   ##    ##     ##   ## ##    ##  ###   ##  ##  ###   ## ##    ##  
   ##    ##     ##  ##   ##   ##  ####  ##  ##  ####  ## ##        
   ##    ########  ##     ##  ##  ## ## ##  ##  ## ## ## ##   #### 
   ##    ##   ##   #########  ##  ##  ####  ##  ##  #### ##    ##  
   ##    ##    ##  ##     ##  ##  ##   ###  ##  ##   ### ##    ##  
   ##    ##     ## ##     ## #### ##    ## #### ##    ##  ######   


def GenWorkSpace(dir,test_percentage,out_dir):

    data_dic = {}
    normpath = os.path.normpath("/".join([dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            file_name = basename.split(".")[0]
            elements_dash = file_name.split("-")
            file_folder = elements_dash[0]
            info = elements_dash[1].split("_scan_Sp")[0].split("_seg_Sp")
            patient = info[0]

            # print(patient)

            if file_folder not in data_dic.keys():
                data_dic[file_folder] = {}

            if patient not in data_dic[file_folder].keys():
                data_dic[file_folder][patient] = {}

            if "_scan" in basename:
                data_dic[file_folder][patient]["scan"] = img_fn

            elif "_seg" in basename:
                data_dic[file_folder][patient]["seg"] = img_fn
            else:
                print("----> Unrecognise CBCT file found at :", img_fn)

    # print(data_dic)
    error = False
    folder_dic = {}
    for folder,patients in data_dic.items():
        if folder not in folder_dic.keys():
            folder_dic[folder] = []
        for patient,data in patients.items():
            if "scan" not in data.keys():
                print("Missing scan for patient :",patient,"at",data["dir"])
                error = True
            if "seg" not in data.keys():
                print("Missing segmentation patient :",patient,"at",data["dir"])
                error = True
            folder_dic[folder].append(data)

    if error:
        print("ERROR : folder have missing/unrecognise files", file=sys.stderr)
        raise


    # print(folder_dic)
    train_data,valid_data = [],[]
    num_patient = 0
    nbr_cv_fold = int(1/test_percentage)
    # print(nbr_cv_fold)
    # nbr_cv_fold = 1
    i = 0
    for i in range(nbr_cv_fold):

        cv_dir_out =  os.path.join(out_dir,"CV_fold_" + str(i))
        if not os.path.exists(cv_dir_out):
            os.makedirs(cv_dir_out)

        data_fold =  os.path.join(cv_dir_out,"data")
        if not os.path.exists(data_fold):
            os.makedirs(data_fold)
        
        patients_fold =  os.path.join(data_fold,"Patients")
        if not os.path.exists(patients_fold):
            os.makedirs(patients_fold)
    
        test_fold =  os.path.join(data_fold,"test")
        if not os.path.exists(test_fold):
            os.makedirs(test_fold)

        for folder,patients in folder_dic.items():

            len_lst = len(patients)
            len_test = int(len_lst/nbr_cv_fold)
            start = i*len_test
            end = (i+1)*len_test
            if end > len_lst: end = len_lst
            training_patients = patients[:start] + patients[end:]
            test_patients = patients[start:end]

            train_cv_dir_out =  os.path.join(patients_fold,folder)
            if not os.path.exists(train_cv_dir_out):
                os.makedirs(train_cv_dir_out)

            for patient in training_patients:
                shutil.copyfile(patient["scan"], os.path.join(train_cv_dir_out,os.path.basename(patient["scan"])))
                shutil.copyfile(patient["seg"], os.path.join(train_cv_dir_out,os.path.basename(patient["seg"])))

            test_cv_dir_out =  os.path.join(test_fold,folder)
            if not os.path.exists(test_cv_dir_out):
                os.makedirs(test_cv_dir_out)

            for patient in test_patients:
                shutil.copyfile(patient["scan"], os.path.join(test_cv_dir_out,os.path.basename(patient["scan"])))
                shutil.copyfile(patient["seg"], os.path.join(test_cv_dir_out,os.path.basename(patient["seg"])))


        # print(training_patients)
        # print(test_patients)



def GetTrainValDataset(dir,val_percentage):
    data_dic = {}
    normpath = os.path.normpath("/".join([dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            file_name = basename.split(".")[0]
            elements_dash = file_name.split("-")
            file_folder = elements_dash[0]
            info = elements_dash[1].split("_scan_Sp")[0].split("_seg_Sp")
            patient = info[0]

            # print(patient)

            if file_folder not in data_dic.keys():
                data_dic[file_folder] = {}

            if patient not in data_dic[file_folder].keys():
                data_dic[file_folder][patient] = {}

            if "_scan" in basename:
                data_dic[file_folder][patient]["scan"] = img_fn

            elif "_seg" in basename:
                data_dic[file_folder][patient]["seg"] = img_fn
            else:
                print("----> Unrecognise CBCT file found at :", img_fn)


    # print(data_dic)
    error = False
    folder_dic = {}
    for folder,patients in data_dic.items():
        if folder not in folder_dic.keys():
            folder_dic[folder] = []
        for patient,data in patients.items():
            if "scan" not in data.keys():
                print("Missing scan for patient :",patient,"at",data["dir"])
                error = True
            if "seg" not in data.keys():
                print("Missing segmentation patient :",patient,"at",data["dir"])
                error = True
            folder_dic[folder].append(data)

    if error:
        print("ERROR : folder have missing/unrecognise files", file=sys.stderr)
        raise


    # print(folder_dic)
    train_data,valid_data = [],[]
    num_patient = 0 
    for folder,patients in folder_dic.items():
        tr,val = train_test_split(patients,test_size=val_percentage,shuffle=True)
        train_data += tr
        valid_data += val
        num_patient += len(patients)

    print("Total patient:", num_patient)

    return train_data,valid_data


########  #######   #######  ##        ######  
   ##    ##     ## ##     ## ##       ##    ## 
   ##    ##     ## ##     ## ##       ##       
   ##    ##     ## ##     ## ##        ######  
   ##    ##     ## ##     ## ##             ## 
   ##    ##     ## ##     ## ##       ##    ## 
   ##     #######   #######  ########  ######  

def MergeSeg(seg_path_dic,out_path,seg_order):
    merge_lst = []
    for id in seg_order:
        if id in seg_path_dic.keys():
            merge_lst.append(seg_path_dic[id])

    first_img = sitk.ReadImage(merge_lst[0])
    main_seg = sitk.GetArrayFromImage(first_img)
    for i in range(len(merge_lst)-1):
        label = i+2
        img = sitk.ReadImage(merge_lst[i+1])
        seg = sitk.GetArrayFromImage(img)
        main_seg = np.where(seg==1,label,main_seg)

    output = sitk.GetImageFromArray(main_seg)
    output.SetSpacing(first_img.GetSpacing())
    output.SetDirection(first_img.GetDirection())
    output.SetOrigin(first_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.Execute(output)
    return output

def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent = 0.95,i_min=-1500, i_max=4000):

    print("Correcting scan contrast :", filepath)
    input_img = sitk.ReadImage(filepath) 
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)


    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)


    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def CloseCBCTSeg(filepath,outpath, closing_radius = 1):
    """
    Close the holes in the CBCT

    Parameters
    ----------
    filePath
     path of the image file 
    radius
     radius of the closing to apply to the seg
    outpath
     path to save the new image
    """

    print("Reading:", filepath)
    input_img = sitk.ReadImage(filepath) 
    img = sitk.GetArrayFromImage(input_img)

    img = np.where(img > 0, 1,img)
    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())

    output = sitk.BinaryDilate(output, [closing_radius] * output.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def ItkToSitk(itk_img):
    new_sitk_img = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_img), isVector=itk_img.GetNumberOfComponentsPerPixel()>1)
    new_sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    new_sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    new_sitk_img.SetDirection(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    return new_sitk_img


def Rescale(filepath,output_spacing=[0.5, 0.5, 0.5]):
    print("Resample :", filepath, ", with spacing :", output_spacing)
    img = itk.imread(filepath)

    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):

        size = itk.size(img)
        scale = spacing/output_spacing

        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()

        #Find new origin
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*spacing
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        VectorImageType = itk.Image[pixel_type, pixel_dimension]
        InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType)
        return resampled_img
        
    else:
        return img
    


def ResampleImage(input,size,spacing,origin,direction,interpolator,IVectorImageType,OVectorImageType):
        ResampleType = itk.ResampleImageFilter[IVectorImageType, OVectorImageType]

        # print(input)

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetInput(input)
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        return resampled_img

def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],interpolator="Linear",outpath=-1):
    """
    Set the spacing of the image at the wanted scale 

    Parameters
    ----------
    filePath
     path of the image file 
    output_spacing
     whanted spacing of the new image file (default : [0.5, 0.5, 0.5])
    outpath
     path to save the new image
    """

    print("Reading:", filepath)
    img = itk.imread(filepath)

    # Dimension = 3
    # InputPixelType = itk.D

    # InputImageType = itk.Image[InputPixelType, Dimension]

    # reader = itk.ImageFileReader[InputImageType].New()
    # reader.SetFileName(filepath)
    # img = reader.GetOutput()

    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):

        size = itk.size(img)
        scale = spacing/output_spacing

        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()

        #Find new origin
        # output_physical_size = np.array(output_size)*np.array(output_spacing)
        # input_physical_size = np.array(size)*spacing
        # output_origin = np.array(input_origin) - (output_physical_size - input_physical_size)/2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        print(pixel_type)

        VectorImageType = itk.Image[pixel_type, pixel_dimension]

        if interpolator == "NearestNeighbor":
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        elif interpolator == "Linear":
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        # print("Already at the wanted spacing")
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img


def SavePrediction(data,ref_filepath, outpath, output_spacing):

    # print("Saving prediction to : ", outpath)

    # print(data)

    ref_img = sitk.ReadImage(ref_filepath) 

    img = data.numpy()[:]

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(output_spacing)
    output.SetDirection(ref_img.GetDirection())
    output.SetOrigin(ref_img.GetOrigin())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)



def CleanScan(file_path):
    input_img = sitk.ReadImage(file_path) 


    closing_radius = 2
    output = sitk.BinaryDilate(input_img, [closing_radius] * input_img.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

    labels_in = sitk.GetArrayFromImage(input_img)
    out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )
    output = sitk.GetImageFromArray(out)
    # closed = sitk.GetArrayFromImage(output)

    # stats = cc3d.statistics(out)
    # mand_bbox = stats['bounding_boxes'][1]
    # rng_lst = []
    # mid_lst = []
    # for slices in mand_bbox:
    #     rng = slices.stop-slices.start
    #     mid = (2/3)*rng+slices.start
    #     rng_lst.append(rng)
    #     mid_lst.append(mid)

    # merge_slice = int(mid_lst[0])
    # out = np.concatenate((out[:merge_slice,:,:],closed[merge_slice:,:,:]),axis=0)
    # output = sitk.GetImageFromArray(out)

    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_path)
    writer.Execute(output)



def SetSpacingFromRef(filepath,refFile,interpolator = "NearestNeighbor",outpath=-1):
    """
    Set the spacing of the image the same as the reference image 

    Parameters
    ----------
    filepath
      image file 
    refFile
     path of the reference image 
    interpolator
     Type of interpolation 'NearestNeighbor' or 'Linear'
    outpath
     path to save the new image
    """

    img = itk.imread(filepath)
    ref = itk.imread(refFile)

    img_sp = np.array(img.GetSpacing()) 
    img_size = np.array(itk.size(img))

    ref_sp = np.array(ref.GetSpacing())
    ref_size = np.array(itk.size(ref))
    ref_origin = ref.GetOrigin()
    ref_direction = ref.GetDirection()

    Dimension = 3
    InputPixelType = itk.D

    InputImageType = itk.Image[InputPixelType, Dimension]

    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(filepath)
    img = reader.GetOutput()

    # reader2 = itk.ImageFileReader[InputImageType].New()
    # reader2.SetFileName(refFile)
    # ref = reader2.GetOutput()

    if not (np.array_equal(img_sp,ref_sp) and np.array_equal(img_size,ref_size)):
        img_info = itk.template(img)[1]
        Ipixel_type = img_info[0]
        Ipixel_dimension = img_info[1]

        ref_info = itk.template(ref)[1]
        Opixel_type = ref_info[0]
        Opixel_dimension = ref_info[1]

        OVectorImageType = itk.Image[Opixel_type, Opixel_dimension]
        IVectorImageType = itk.Image[Ipixel_type, Ipixel_dimension]

        if interpolator == "NearestNeighbor":
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[InputImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        elif interpolator == "Linear":
            InterpolatorType = itk.LinearInterpolateImageFunction[InputImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,ref_size.tolist(),ref_sp,ref_origin,ref_direction,interpolator,InputImageType,InputImageType)

        output = ItkToSitk(resampled_img)
        output = sitk.Cast(output, sitk.sitkInt16)

        if img_sp[0] > ref_sp[0]:
            closing_radius = 2
            MedianFilter = sitk.BinaryMedianImageFilter()
            MedianFilter.SetRadius(closing_radius)
            output = MedianFilter.Execute(output)


        if outpath != -1:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(outpath)
            writer.Execute(output)
                # itk.imwrite(resampled_img, outpath)
        return output

    else:
        output = ItkToSitk(img)
        output = sitk.Cast(output, sitk.sitkInt16)
        if outpath != -1:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(outpath)
            writer.Execute(output)
        return output

def KeepLabel(filepath,outpath,labelToKeep):

    # print("Reading:", filepath)
    input_img = sitk.ReadImage(filepath) 
    img = sitk.GetArrayFromImage(input_img)

    for i in range(np.max(img)):
        label = i+1
        if label != labelToKeep:
            img = np.where(img == label, 0,img)

    img = np.where(img > 0, 1,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def ConvertSimpleItkImageToItkImage(_sitk_image: sitk.Image, _pixel_id_value):
    """
    Converts SimpleITK image to ITK image
    :param _sitk_image: SimpleITK image
    :param _pixel_id_value: Type of the pixel in SimpleITK format (for example: itk.F, itk.UC)
    :return: ITK image
    """
    array: np.ndarray = sitk.GetArrayFromImage(_sitk_image)
    itk_image: itk.Image = itk.GetImageFromArray(array)
    itk_image = CopyImageMetaInformationFromSimpleItkImageToItkImage(itk_image, _sitk_image, _pixel_id_value)
    return itk_image

def CopyImageMetaInformationFromSimpleItkImageToItkImage(_itk_image: itk.Image, _reference_sitk_image: sitk.Image, _output_pixel_type) -> itk.Image:
    """
	Copies the meta information from SimpleITK image to ITK image
    :param _itk_image: Source ITK image
    :param _reference_sitk_image: Original SimpleITK image from which will be copied the meta information
    :param _pixel_type: Type of the pixel in SimpleITK format (for example: itk.F, itk.UC)
    :return: ITK image with the new meta information
    """
    _itk_image.SetOrigin(_reference_sitk_image.GetOrigin())
    _itk_image.SetSpacing(_reference_sitk_image.GetSpacing())

    # Setting the direction (cosines of the study coordinate axis direction in the space)
    reference_image_direction: np.ndarray = np.eye(3)
    np_dir_vnl = itk.GetVnlMatrixFromArray(reference_image_direction)
    itk_image_direction = _itk_image.GetDirection()
    itk_image_direction.GetVnlMatrix().copy_in(np_dir_vnl.data_block())

    dimension: int = _itk_image.GetImageDimension()
    input_image_type = type(_itk_image)
    output_image_type = itk.Image[_output_pixel_type, dimension]

    castImageFilter = itk.CastImageFilter[input_image_type, output_image_type].New()
    castImageFilter.SetInput(_itk_image)
    castImageFilter.Update()
    result_itk_image: itk.Image = castImageFilter.GetOutput()

    return result_itk_image

def PlotState(img,label,x,y,z):
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    plt.figure("scan", (18, 6))
    plt.subplot(3, 2, 1)
    plt.title("scan")
    plt.imshow(img[0, :, :, z].detach().cpu(), cmap="gray")
    plt.subplot(3, 2, 2)
    plt.title("seg")
    plt.imshow(label[0, :, :, z].detach().cpu())
    plt.subplot(3, 2, 3)
    plt.imshow(img[0, :, y, :].detach().cpu(), cmap="gray")
    plt.subplot(3, 2, 4)
    plt.imshow(label[0, :, y, :].detach().cpu())
    plt.subplot(3, 2, 5)
    plt.imshow(img[0, x, :, :].detach().cpu(), cmap="gray")
    plt.subplot(3, 2, 6)
    plt.imshow(label[0, x, :, :].detach().cpu())
    plt.show()

