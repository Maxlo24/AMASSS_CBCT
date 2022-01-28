from models import*
from utils import*

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
            if "_scan" in basename:
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

    with torch.no_grad():
        for data in data_list:

            pred_img,input_img = CreatePredictTransform(data)
            # print(pred_img.size())
            val_inputs = pred_img.unsqueeze(0)
            # print(val_inputs.size())
            val_outputs = val_inputs
            val_outputs = sliding_window_inference(
                inputs= val_inputs.to(DEVICE),
                roi_size = cropSize, 
                sw_batch_size= nbr_workers, 
                predictor= net, 
                overlap=0.25
            )

            out_img = torch.argmax(val_outputs, dim=1).detach().cpu()
            out_img = out_img.type(torch.int16)
            # print(out_img,np.shape(out_img))

            baseName = os.path.basename(data)
            scan_name= baseName.split(".")
            pred_name = ""
            for i,element in enumerate(scan_name):
                if i == 0:
                    pred_name += element.replace("scan","Pred")
                else:
                    pred_name += "." + element

            input_dir = os.path.dirname(data)
            
            SavePrediction(out_img ,input_img,os.path.join(input_dir,pred_name))
            
    print("Done : " + str(len(data_list)) + " scan segmented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--load_model', type=str, help='Path of the model', default=None, required=True)
    # input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)

    
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.5,0.5,0.5])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=2)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=1)

    args = parser.parse_args()
    
    main(args)
