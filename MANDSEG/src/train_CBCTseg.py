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

    train_transforms = CreateTrainTransform(cropSize,1,10)
    val_transforms = CreateValidationTransform()

    trainingSet,validationSet = GetTrainValDataset(args.dir_patients,args.test_percentage/100)

    print(validationSet)
    model = Create_UNETR(
        input_channel=1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(DEVICE)

    # model.load_state_dict(torch.load("/Users/luciacev-admin/Documents/Projects/Benchmarks/CBCT_Seg_benchmark/data/best_model.pth",map_location=DEVICE))



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
    
    # case_num = 0
    # img = val_ds[case_num]["scan"]
    # label = val_ds[case_num]["seg"]
    # size = img.shape
    # PlotState(img,label,int(size[1]/2),int(size[2]/2),int(size[1]/3.5))
    # for i,data in enumerate(train_ds[case_num]):
    #     img = data["scan"]
    #     label = data["seg"]
    #     size = img.shape
    #     PlotState(img,label,int(size[1]/2),int(size[2]/2),int(size[1]/2))

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
    # TM.Validate()
    TM.Process(args.max_epoch)

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

        self.predictor = 10

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
            x, y = self.RandomPermutChannels(x,y)
            # print(x.shape,x.dtype,y.shape,y.dtype)
            logit_map = self.model(x)
            # print(logit_map.shape,logit_map.dtype)
            loss = self.loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (loss=%2.5f)" % (loss)
            )
        mean_loss = epoch_loss/steps
        self.loss_lst.append(mean_loss)
        self.tensorboard.add_scalar("Training loss",mean_loss,self.epoch)
        self.tensorboard.close()



    def Validate(self):
        self.model.eval()
        dice_vals = list()
        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (dice=X.X)", dynamic_ncols=True
        )
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["scan"].to(self.device), batch["seg"].to(self.device))
                val_inputs, val_labels = self.RandomPermutChannels(val_inputs,val_labels)

                # print("IN INFO")
                # print(val_inputs)
                # print(torch.min(val_inputs),torch.max(val_inputs))
                # print(val_inputs.shape)
                # print(val_inputs.dtype)

                val_outputs = sliding_window_inference(val_inputs, self.FOV, self.predictor, self.model,overlap=0.2)
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
                # self.SaveScans(val_inputs,val_outputs,step)
            self.dice_metric.reset()


        mean_dice_val = np.mean(dice_vals)
        self.dice_lst.append(mean_dice_val)

        if mean_dice_val > self.best_dice:
            torch.save(self.model.state_dict(), os.path.join(self.save_model_dir,"best_model.pth"))
            print("Model Was Saved ! Current Best Avg. Dice: {} Previous Best Avg. Dice: {}".format(mean_dice_val, self.best_dice))
            self.best_dice = mean_dice_val
        else:
            print("Model Was Not Saved ! Best Avg. Dice: {} Current Avg. Dice: {}".format(self.best_dice, mean_dice_val))

        self.tensorboard.add_scalar("Validation dice",mean_dice_val,self.epoch)

        self.PrintSlices(val_inputs,val_labels,val_outputs)
        self.tensorboard.close()
    
    def RandomPermutChannels(self,batch,batch2):
        prob = np.random.rand()
        if prob < 0.25:
            permImg = batch.permute(0,1,2,4,3)
            permImg2 = batch2.permute(0,1,2,4,3)
        elif prob < 0.50:
            permImg = batch.permute(0,1,4,3,2)
            permImg2 = batch2.permute(0,1,4,3,2)
        elif prob < 0.75:
            permImg = batch.permute(0,1,3,2,4)
            permImg2 = batch2.permute(0,1,3,2,4)
        else:
            permImg = batch
            permImg2 = batch2
        return permImg,permImg2
        
    def PrintSlices(self,val_inputs,val_labels,val_outputs):

        size = val_inputs.shape[4]
        seg = torch.argmax(val_outputs, dim=1).detach()

        inpt_lst = []
        lab_lst = []
        seg_lst = []
        for slice in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            slice_nbr = int(size*slice)

            inpt_lst.append(val_inputs.cpu()[0, 0, :, :, slice_nbr].unsqueeze(0))
            lab_lst.append(val_labels.cpu()[0, 0, :, :, slice_nbr].unsqueeze(0))
            seg_lst.append(seg.cpu()[0, :, :, slice_nbr].unsqueeze(0))

        img_lst = inpt_lst + lab_lst + seg_lst
        slice_view = torch.cat(img_lst,dim=0).unsqueeze(1)
        self.tensorboard.add_images("Validation images",slice_view,self.epoch)

    def SaveScans(self,val_inputs,val_outputs,step):

        data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)
        print(data.shape)
        img = data.numpy()[0][:]
        output = sitk.GetImageFromArray(img)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(step)+'_seg.nii.gz')
        writer.Execute(output)

        img = val_inputs.squeeze(0).numpy()[0][:]
        output = sitk.GetImageFromArray(img)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(step)+'_scan.nii.gz')
        writer.Execute(output)





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
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=13)
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    input_group.add_argument('-me', '--max_epoch', type=int, help='Number of training epocs', default=250)
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-bs', '--batch_size', type=int, help='batch size', default=2)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)



    args = parser.parse_args()
    
    main(args)