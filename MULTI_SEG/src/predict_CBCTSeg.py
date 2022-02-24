from models import*
from utils import*
import time
import os
import shutil


from monai.data import (
    DataLoader,
    Dataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import argparse

def main(args):

    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker
    spacing = args.spacing
    cropSize = args.crop_size

    temp_fold = args.temp_fold

    print("Loading data from", args.dir)


    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)

    data_list = []
    normpath = os.path.normpath("/".join([args.dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                # new_path = os.path.join(temp_fold,basename)
                # CorrectHisto(img_fn, new_path,0.01, 0.99)
                data_list.append({"scan":img_fn, "name":img_fn, "temp_path":img_fn})



    net = Create_UNETR(
        input_channel = 1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(DEVICE)

    print("Loading model", args.load_model)
    net.load_state_dict(torch.load(args.load_model,map_location=DEVICE))
    # net.eval()
    # net.double()

    # define pre transforms
    # pre_transforms = createTestTransform(wanted_spacing= args.spacing,outdir=args.out)

    pred_transform = CreatePredTransform()


    pred_ds = Dataset(
        data=data_list, 
        transform=pred_transform, 
    )
    pred_loader = DataLoader(
        dataset=pred_ds,
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )


    startTime = time.time()

    net.eval()

    # pred_iterator = tqdm(
    #     pred_loader, desc="Pred", dynamic_ncols=True
    # )
    with torch.no_grad():
        for step, batch in enumerate(pred_loader):
            input_img, input_path,temp_path = (batch["scan"].to(DEVICE), batch["name"][0],batch["temp_path"][0])
            print(input_path)
            # val_outputs = sliding_window_inference(input_img, cropSize, nbr_workers, net,overlap=args.precision)

            # pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)

            baseName = os.path.basename(input_path)
            scan_name= baseName.split(".")
            # print(baseName)
            if "_scan" in baseName:
                pred_name = baseName.replace("_scan","_Pred")
            elif "_Scan" in baseName:
                pred_name = baseName.replace("_Scan","_Pred")
            else:
                pred_name = ""
                for i,element in enumerate(scan_name):
                    if i == 0:
                        pred_name += element + "_Pred"
                    else:
                        pred_name += "." + element

            input_dir = os.path.dirname(input_path)
            file_path = os.path.join(input_dir,pred_name)

            input_img_no_batch = input_img.squeeze(0)
            input_img = input_img_no_batch

            # input_img = input_img_no_batch.permute(0,3,2,1)
            # print(input_img.shape)
            # SavePrediction(input_img,input_path,os.path.join(input_dir,scan_name[0] + "_RESCALE.nii.gz"))
            # SavePrediction(input_img,input_path,file_path)

            # SetSpacing(input_path,[0.5,0.5,0.5],file_path)
            
            # SavePrediction(pred_data.permute(0,3,2,1),input_path,temp_path)
            # CleanScan(temp_path)
            # SetSpacingFromRef(
            #     temp_path,
            #     input_path,
            #     # "Linear",
            #     outpath=file_path
            #     )

            # vtk_path = file_path.split(".")[0] + ".vtp"
            # SavePredToVTK(file_path,vtk_path)

            

    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))

    stopTime = time.time()
    print("Done : " + str(len(data_list)) + " scan segmented in :", stopTime-startTime, "seconds")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans', default='/Users/luciacev-admin/Documents/Projects/Benchmarks/CBCT_Seg_benchmark/data/test')
    input_group.add_argument('--load_model', type=str, help='Path of the model', default='/Users/luciacev-admin/Documents/Projects/Benchmarks/CBCT_Seg_benchmark/data/best_model_MAND.pth')
    # input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)
    input_group.add_argument('--temp_fold', type=str, help='temporary folder', default='../temp')
    

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.5,0.5,0.5])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    input_group.add_argument('-p', '--precision', type=float, help='precision of the prediction', default=0.2)

    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=2)

    args = parser.parse_args()
    
    main(args)