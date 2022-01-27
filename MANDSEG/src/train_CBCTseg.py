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

from torch.utils.tensorboard import SummaryWriter


def main(args):

    # #####################################
    #  Init_param
    # #####################################
    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker

    cropSize = args.crop_size

    train_transforms = CreateTrainTransform(cropSize,10,2)
    val_transforms = CreateValidationTransform()

    trainingSet,validationSet = GetTrainValDataset(args.dir_patients,args.test_percentage/100)

    print(validationSet)
    model = Create_UNETR(
        input_channel=1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(DEVICE)

    torch.backends.cudnn.benchmark = True

    train_ds = CacheDataset(
        data=trainingSet,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=nbr_workers,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, 
        shuffle=True,
        num_workers=nbr_workers, 
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
        num_workers=nbr_workers, 
        pin_memory=True
    )
    
    case_num = 0
    # img = val_ds[case_num]["scan"]
    # label = val_ds[case_num]["seg"]
    # PlotState(img,label,40,40,15)
    # for i,data in enumerate(train_ds[case_num]):
    #     img = data["scan"]
    #     label = data["seg"]
    #     PlotState(img,label,32,32,32)

    TM = TrainingMaster(
        model = model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_model_dir=args.dir_model,
        save_runs_dir=args.dir_data,
        nbr_label = label_nbr,
        FOV=cropSize,
        device=DEVICE
        )

    # TM.Train()
    TM.Validate()
    # TM.Process(10)


class TrainingMaster:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        save_model_dir,
        save_runs_dir,
        nbr_label = 2,
        FOV = [64,64,64],
        device = DEVICE,
        ) -> None:
        self.model = model
        self.device = device
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.post_label = AsDiscrete(to_onehot=True,num_classes=nbr_label)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True,num_classes=nbr_label)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

        self.save_model_dir = save_model_dir
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

        run_path = save_runs_dir + "/Runs"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        self.tensorboard = SummaryWriter(run_path)

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.FOV = FOV

        self.epoch = 0
        self.best_dice = 0
        self.loss_lst = []
        self.dice_lst = []

    def Process(self,num_epoch):
        for epoch in range(num_epoch):
            self.Train()
            self.Validate()
            self.epoch += 1
            self.tensorboard.close()

    def Train(self):
        self.model.train()
        epoch_loss = 0
        steps = 0
        epoch_iterator = tqdm(
            self.train_loader, desc="Training (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            steps += 1
            x, y = (batch["scan"].to(self.device), batch["seg"].to(self.device))
            print(x.shape,x.dtype,y.shape,y.dtype)
            logit_map = self.model(x)
            print(logit_map.shape,logit_map.dtype)
            loss = self.loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (loss=%2.5f)" % (loss)
            )

        self.loss_lst.append(epoch_loss/steps)

        self.tensorboard.add_scalar("Training loss",epoch_loss,self.epoch)

    def Validate(self):
        self.model.eval()
        dice_vals = list()
        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (dice=X.X)", dynamic_ncols=True
        )
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["scan"].to(self.device), batch["seg"].to(self.device))
                val_outputs = sliding_window_inference(val_inputs, self.FOV, 4, self.model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    self.post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                self.dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = self.dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (dice=%2.5f)" % (dice)
                )
            self.dice_metric.reset()

        mean_dice_val = np.mean(dice_vals)
        self.dice_lst.append(mean_dice_val)

        if mean_dice_val > self.best_dice:
            torch.save(self.model.state_dict(), os.path.join(self.save_model_dir,"best_model.pth"))
            print("Model Was Saved ! Current Best Avg. Dice: {} Previous Best Avg. Dice: {}".format(mean_dice_val, self.best_dice))
            self.best_dice = mean_dice_val
        else:
            print("Model Was Not Saved ! Best Avg. Dice: {} Current Avg. Dice: {}".format(self.best_dice, mean_dice_val))

        input_slice = val_inputs.cpu()[0, 0, :, :, 20].unsqueeze(0)
        labels_slice = val_labels.cpu()[0, 0, :, :, 20].unsqueeze(0)
        seg_slice = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 20].unsqueeze(0)
        input_slice1 = val_inputs.cpu()[0, 0, :, :, 25].unsqueeze(0)
        labels_slice1 = val_labels.cpu()[0, 0, :, :, 25].unsqueeze(0)
        seg_slice1 = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 25].unsqueeze(0)
        input_slice2 = val_inputs.cpu()[0, 0, :, :, 30].unsqueeze(0)
        labels_slice2 = val_labels.cpu()[0, 0, :, :, 30].unsqueeze(0)
        seg_slice2 = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 30].unsqueeze(0)
        slice_view = torch.cat((input_slice,labels_slice,seg_slice,input_slice1,labels_slice1,seg_slice1,input_slice2,labels_slice2,seg_slice2),dim=0).unsqueeze(1)
        self.tensorboard.add_images("Validation images",slice_view,self.epoch)
        self.tensorboard.add_scalar("Validation dice",mean_dice_val,self.epoch)


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
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=10)
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='Number of training epocs', default=250)
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-bs', '--batch_size', type=int, help='batch size', default=2)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)



    args = parser.parse_args()
    
    main(args)