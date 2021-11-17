import numpy as np  
import SimpleITK as sitk
import itk
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

import datetime
import glob
import sys

# ----- MONAI ------

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AddChannel,
    Compose,
    CropForegroundd,
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
    RandRotate90d,
    ToTensord,
    ToTensor,
    SaveImaged,
    SaveImage,
    RandCropByLabelClassesd,
    Lambdad,
    CastToTyped,
    SpatialCrop,
    BorderPad
)

from monai.config import print_config
from monai.metrics import DiceMetric

from monai.data import (
    DataLoader,
    CacheDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



######## ########     ###    ##    ##  ######  ########  #######  ########  ##     ##  ######  
   ##    ##     ##   ## ##   ###   ## ##    ## ##       ##     ## ##     ## ###   ### ##    ## 
   ##    ##     ##  ##   ##  ####  ## ##       ##       ##     ## ##     ## #### #### ##       
   ##    ########  ##     ## ## ## ##  ######  ######   ##     ## ########  ## ### ##  ######  
   ##    ##   ##   ######### ##  ####       ## ##       ##     ## ##   ##   ##     ##       ## 
   ##    ##    ##  ##     ## ##   ### ##    ## ##       ##     ## ##    ##  ##     ## ##    ## 
   ##    ##     ## ##     ## ##    ##  ######  ##        #######  ##     ## ##     ##  ######  


def CreateTrainTransform(CropSize = [64,64,64]):
    train_transforms = Compose(
        [
            LoadImaged(keys=["scan", "seg"]),
            AddChanneld(keys=["scan", "seg"]),
            ScaleIntensityd(
                keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
            ),
            RandCropByPosNegLabeld(
                keys=["scan", "seg"],
                label_key="seg",
                spatial_size=CropSize,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="scan",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["scan", "seg"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["scan", "seg"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["scan", "seg"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["scan", "seg"],
                prob=0.10,
                max_k=3,
            ),
            # RandShiftIntensityd(
            #     keys=["scan"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),
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
            ToTensord(keys=["scan", "seg"]),
        ]
    )
    return val_transforms


######## ########     ###    #### ##    ## #### ##    ##  ######   
   ##    ##     ##   ## ##    ##  ###   ##  ##  ###   ## ##    ##  
   ##    ##     ##  ##   ##   ##  ####  ##  ##  ####  ## ##        
   ##    ########  ##     ##  ##  ## ## ##  ##  ## ## ## ##   #### 
   ##    ##   ##   #########  ##  ##  ####  ##  ##  #### ##    ##  
   ##    ##    ##  ##     ##  ##  ##   ###  ##  ##   ### ##    ##  
   ##    ##     ## ##     ## #### ##    ## #### ##    ##  ######   
 
def train(inID, outID, data_model, global_step, epoch_loss_values, max_iterations, train_loader ):
    model = data_model["model"]
    model.train()
    epoch_loss = 0
    steps = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        steps += 1

        input = torch.cat([batch[key] for key in inID],1)
        # print(input.size())
        x, y = (input.to(DEVICE), batch[outID].to(DEVICE))
        # print(y.shape)
        logit_map = model(x)
        # print(logit_map.shape)
        loss = data_model["loss_f"](logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        data_model["optimizer"].step()
        data_model["optimizer"].zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+steps, max_iterations, loss)
        )
        data_model["model"] = model
    epoch_loss /= steps
    epoch_loss_values.append(epoch_loss)

    return steps

def validation(inID, outID,model,cropSize, post_label, post_pred, dice_metric, global_step, epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            input = torch.cat([batch[key] for key in inID],1)
            val_inputs, val_labels = (input.to(DEVICE), batch[outID].to(DEVICE))
            val_outputs = sliding_window_inference(val_inputs, cropSize, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            # dice = IoU_Dice(y_true_lst=val_output_convert, y_pred_lst=val_labels_convert)
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)

    return mean_dice_val


def validate(inID, outID,data_model,val_loader, cropSize, global_step, metric_values, dice_val_best, global_step_best, dice_metric, post_label, post_pred):
    model = data_model["model"]
    epoch_loss = 0
    step = 0

    epoch_iterator_val = tqdm(
        val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    )
    dice_val = validation(
        inID=inID,
        outID = outID,
        model=model,
        cropSize=cropSize,
        global_step=global_step,
        epoch_iterator_val=epoch_iterator_val,
        dice_metric=dice_metric,
        post_label=post_label,
        post_pred=post_pred
    )
    metric_values.append(dice_val)
    if dice_val > dice_val_best:
        dice_val_best = dice_val
        global_step_best = global_step
        save_path = os.path.join(data_model["dir"],data_model["name"]+"_"+datetime.datetime.now().strftime("%Y_%d_%m")+"_E_"+str(global_step)+".pth")
        torch.save(
            model.state_dict(), save_path
        )
        data_model["best"] = save_path
        print("Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
    else:
        print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
    
    # global_step += 1
    data_model["model"] = model

    return dice_val_best, global_step_best



########  #######   #######  ##        ######  
   ##    ##     ## ##     ## ##       ##    ## 
   ##    ##     ## ##     ## ##       ##       
   ##    ##     ## ##     ## ##        ######  
   ##    ##     ## ##     ## ##             ## 
   ##    ##     ## ##     ## ##       ##    ## 
   ##     #######   #######  ########  ######  

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
    for folder,patients in folder_dic.items():
        tr,val = train_test_split(patients,test_size=val_percentage,shuffle=True)
        train_data += tr
        valid_data += val

    return train_data,valid_data


def CorrectCBCT(filepath,outpath, closing_radius = 5):
    """
    Remove the unwanted labels from a file and make the other one bigger  

    Parameters
    ----------
    filePath
     path of the image file 
    labelToRemove
     list of the labels to remove from the image 
    label_radius
     radius of the dilatation to apply to the remaining labels
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
    output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def ResampleImage(input,size,spacing,origin,direction,interpolator,VectorImageType):
        ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.SetInput(input)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        return resampled_img

def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],outpath=-1):
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

        if True in [seg in os.path.basename(filepath) for seg in ["seg","Seg"]]:
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        else:
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        # print("Already at the wanted spacing")
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img

