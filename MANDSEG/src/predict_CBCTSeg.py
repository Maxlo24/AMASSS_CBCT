from models import*
from utils import*
import time

import argparse

def main(args):

    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker
    spacing = args.spacing
    cropSize = args.crop_size

    data_list = []
    normpath = os.path.normpath("/".join([args.dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            if "_Pred" not in basename:
                data_list.append(img_fn)


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


    print("Loading data from", args.dir)

    startTime = time.time()

    net.eval()

    with torch.no_grad():
        for data in data_list:

            input_img,ref_img = CreatePredictTransform(data,args.spacing[0])

            print(input_img.size())
            rescaled_img = input_img
            input_img = input_img.permute(0,2,3,1)
            val_inputs = input_img.unsqueeze(0)
            # print("IN INFO")
            # print(val_inputs)
            # print(torch.min(val_inputs),torch.max(val_inputs))
            # print(val_inputs.shape)
            # print(val_inputs.dtype)
            # print(val_inputs.size())
            val_outputs = sliding_window_inference(val_inputs, cropSize, nbr_workers, net,overlap=0.25)

            pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)
            pred_data = pred_data.permute(0,3,1,2)


            baseName = os.path.basename(data)
            scan_name= baseName.split(".")
            print(baseName)
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

            input_dir = os.path.dirname(data)
            file_path = os.path.join(input_dir,pred_name)

            SavePrediction(input_img,ref_img,os.path.join(input_dir,scan_name[0] + "_RESCALE.nii.gz"))
            SavePrediction(pred_data,ref_img,file_path)


    stopTime = time.time()
    print("Done : " + str(len(data_list)) + " scan segmented in :", stopTime-startTime, "seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans', default='/Users/luciacev-admin/Documents/Projects/Benchmarks/CBCT_Seg_benchmark/data/test')
    input_group.add_argument('--load_model', type=str, help='Path of the model', default='/Users/luciacev-admin/Documents/Projects/Benchmarks/CBCT_Seg_benchmark/data/best_model.pth')
    # input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)
    
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.5,0.5,0.5])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=1)

    args = parser.parse_args()
    
    main(args)
