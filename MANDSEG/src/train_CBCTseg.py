import torch
from models import *
from utils import *

import argparse

import logging
import sys

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

from monai.data import (
    DataLoader,
    CacheDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

def main(args):

    # #####################################
    #  Init_param
    # #####################################
    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker

    cropSize = args.crop_size

    train_transforms = CreateTrainTransform(cropSize)
    val_transforms = CreateValidationTransform()

    trainingSet,validationSet = GetTrainValDataset(args.dir_patients,args.test_percentage/100)

    print(validationSet)
    model = Create_UNETR(
        input_channel=1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(DEVICE)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_ds = CacheDataset(
        data=trainingSet,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=nbr_workers,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, 
        shuffle=True, num_workers=8, 
        pin_memory=True
    )
    val_ds = CacheDataset(
        data=validationSet, 
        transform=val_transforms, 
        cache_rate=1.0, 
        num_workers=nbr_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    case_num = 1
    img = val_ds[case_num]["scan"]
    label = val_ds[case_num]["seg"]
    PlotState(img,label,40,40,15)
    for i,data in enumerate(train_ds[case_num]):
        img = data["scan"]
        label = data["seg"]
        PlotState(img,label,32,32,32)



# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training to find ROI for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/Benchmarks/CBCT_Seg_benchmark')
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_patients', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_data+'/Patients')
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Models')

    input_group.add_argument('-mn', '--model_name', type=str, help='Name of the model', default="MandSeg_model")
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='Number of training epocs', default=250)
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-bs', '--batch_size', type=int, help='batch size', default=10)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)



    args = parser.parse_args()
    
    main(args)