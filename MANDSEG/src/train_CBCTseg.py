from models import *
from utils import *

import argparse

import logging
import sys



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

    # #####################################
    #  Load data
    # #####################################

    # print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # print(trainingSet)

    train_ds = CacheDataset(
        data=trainingSet,
        transform=train_transforms,
        # cache_num=24,
        cache_rate=1.0,
        num_workers=nbr_workers,
    )
    # print(len(train_ds[0]))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nbr_workers, pin_memory=True
    )
  
    val_ds = CacheDataset(
        data=validationSet,
        transform=val_transforms, 
        # cache_num=6, 
        cache_rate=1.0, 
        num_workers=nbr_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=nbr_workers, pin_memory=True
    )

    # #####################################
    #  Training
    # #####################################

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    model = Create_UNETR(
        input_channel=1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(DEVICE)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model_data = {
        "model" : model,
        "name": args.model_name,
        "dir":args.dir_model, 
        "loss_f":loss_function, 
        "optimizer":optimizer 
        }

    max_iterations = args.max_iterations
    eval_num = 500
    post_label = AsDiscrete(to_onehot=True, n_classes=label_nbr)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=label_nbr)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    step_to_val = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        if (step_to_val >= eval_num) or global_step >= max_iterations:
            dice_val_best, global_step_best = validate(
                inID=["scan"],
                outID = "seg",
                data_model=model_data,
                val_loader = val_loader,
                cropSize=cropSize,
                global_step=global_step,
                metric_values=metric_values,
                dice_val_best=dice_val_best,
                global_step_best=global_step_best,
                dice_metric=dice_metric,
                post_pred=post_pred,
                post_label=post_label,
            )
            step_to_val -= eval_num
            global_step +=1
        steps = train(
            inID=["scan"],
            outID = "seg",
            data_model=model_data,
            global_step=global_step,
            epoch_loss_values=epoch_loss_values,
            max_iterations=max_iterations,
            train_loader=train_loader,
        )
        global_step += steps
        step_to_val += steps

    print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
    )
    print("Best model at : ", model_data["best"])


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
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='Number of training epocs', default=25000)
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-bs', '--batch_size', type=int, help='batch size', default=10)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)



    args = parser.parse_args()
    
    main(args)