from models import*
from utils import*
import time
import os
import shutil
import random
import string

#generate random id
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

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
    spacing = args.spacing
    cropSize = args.crop_size

    temp_fold = os.path.join(args.temp_fold, "temp_" + id_generator())
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)

    model_dict = {}
    print("Loading models from", args.dir_models)
    normpath = os.path.normpath("/".join([args.dir_models, '**', '']))
    for img_fn in glob.iglob(normpath, recursive=True):
        #  print(img_fn)
        basename = os.path.basename(img_fn)
        if basename.endswith(".pth"):
            model_id = basename.split("_")[1]
            model_dict[model_id] = img_fn

    # load data
    data_list = []

    if args.file:
        print("Loading scan :", args.file)
        img_fn = args.file
        basename = os.path.basename(img_fn)
        new_path = os.path.join(temp_fold,basename)
        temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
        CorrectHisto(img_fn, new_path,0.01, 0.99)
        # new_path = img_fn
        data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})

    else:
        scan_dir = args.dir
        print("Loading data from",scan_dir )
        normpath = os.path.normpath("/".join([scan_dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    new_path = os.path.join(temp_fold,basename)
                    temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
                    CorrectHisto(img_fn, new_path,0.01, 0.99)
                    data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})

    net = Create_UNETR(
        input_channel = 1,
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(DEVICE)

    # net.eval()
    # net.double()

    # define pre transforms
    # pre_transforms = createTestTransform(wanted_spacing= args.spacing,outdir=args.out)

    pred_transform = CreatePredTransform(spacing)

    pred_ds = Dataset(
        data=data_list, 
        transform=pred_transform, 
    )
    pred_loader = DataLoader(
        dataset=pred_ds,
        batch_size=1, 
        shuffle=False, 
        num_workers=args.nbr_CPU_worker, 
        pin_memory=True
    )

    startTime = time.time()
    # pred_iterator = tqdm(
    #     pred_loader, desc="Pred", dynamic_ncols=True
    # )

    if args.skul_structure[0] != "ALL":
        temp_dic = {}
        for id in args.skul_structure:
            temp_dic[id] = model_dict[id]
        model_dict = temp_dic


    with torch.no_grad():
        for step, batch in enumerate(pred_loader):
            input_img, input_path,temp_path = (batch["scan"].to(DEVICE), batch["name"],batch["temp_path"])


            image = input_path[0]
            print("Working on :",image)
            baseName = os.path.basename(image)
            scan_name= baseName.split(".")
            # print(baseName)
            pred_id = "_XXXX-Seg_Pred"

            if "_scan" in baseName:
                pred_name = baseName.replace("_scan",pred_id)
            elif "_Scan" in baseName:
                pred_name = baseName.replace("_Scan",pred_id)
            else:
                pred_name = ""
                for i,element in enumerate(scan_name):
                    if i == 0:
                        pred_name += element + pred_id
                    else:
                        pred_name += "." + element

            merge_dic_list={}


            if args.save_in_folder:
                outputdir = os.path.dirname(input_path[0]) + "/" + scan_name[0] + "_" + "SegOut"

                print("Output dir :",outputdir)

                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
            else :
                outputdir = os.path.dirname(input_path[0])


            for model_id,model_path in model_dict.items():
                print("Loading model", model_path)
                net.load_state_dict(torch.load(model_path,map_location=DEVICE))
                net.eval()

                val_outputs = sliding_window_inference(input_img, cropSize, args.nbr_GPU_worker, net,overlap=args.precision)

                pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)


                # input_img_no_batch = input_img.squeeze(0)
                # input_img = input_img_no_batch

                # input_img = input_img_no_batch.permute(0,3,2,1)
                # print(input_img.shape)
                # SavePrediction(input_img,input_path,os.path.join(input_dir,scan_name[0] + "_RESCALE.nii.gz"))
                # SavePrediction(input_img,input_path,file_path)

                # SetSpacing(input_path,[0.5,0.5,0.5],file_path)

                segmentations = pred_data.permute(0,3,2,1)

                seg = segmentations.squeeze(0)



                file_path = os.path.join(outputdir,pred_name.replace('XXXX',model_id))
                SavePrediction(seg,input_path[0],temp_path[0],output_spacing = spacing)
                # if not True in [model_id == id for id in ["CV","RC","FF"]]:
                #     CleanScan(temp_path[0])
                SetSpacingFromRef(
                    temp_path[0],
                    input_path[0],
                    # "Linear",
                    outpath=file_path
                    )
                merge_dic_list[model_id] = file_path

                if args.gen_vtk:
                    SavePredToVTK(file_path,temp_fold, args.vtk_smooth)

            if args.merge and len(merge_dic_list.keys()) > 1:


                print("Merging")
                outpath = os.path.join(outputdir,pred_name.replace('XXXX','MERGED'))

                MergeSeg(merge_dic_list,outpath,args.merging_order)
                # merge_dic_list["MERGED"] = outpath


    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))

    print("Done in %.2f seconds" % (time.time() - startTime))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #Generate mutually exclusive group
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('-id','--dir', type=str, help='Path to the scans folder', default='/app/data/scans')
    # group.add_argument('-if','--file', type=str, help='Path to the scan', default=None)
    # input_group.add_argument('--dir', type=str, help='Input directory with the scans', default='/app/data/scans')
    
    input_group = parser.add_argument_group('directory')

    input_group.add_argument('-id','--dir', type=str, help='Path to the scans folder', default='/app/data/scans')
    input_group.add_argument('-if','--file', type=str, help='Path to the scan', default=None)
    input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models', default='/home/luciacev/Desktop/Maxime_Gillot/Data/AMASSS/FULL_FACE_MODELS')
    # input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models', default='/app/data/ALL_MODELS')
    # input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)
    input_group.add_argument('--temp_fold', type=str, help='temporary folder', default='..')

    input_group.add_argument('-ss', '--skul_structure', nargs="+", type=str, help='Skul structure to segment', default=["FF"])
    input_group.add_argument('-m', '--merge', type=bool, help='merge the segmentations', default=False)
    input_group.add_argument('-sf', '--save_in_folder', type=bool, help='Save the output in one folder', default=False)

    input_group.add_argument('-vtk', '--gen_vtk', type=bool, help='Genrate vtk file', default=True)
    input_group.add_argument('--vtk_smooth', type=int, help='Smoothness of the vtk', default=5)


    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.4,0.4,0.4])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    input_group.add_argument('-pr', '--precision', type=float, help='precision of the prediction', default=0.5)

    input_group.add_argument('-mo','--merging_order',nargs="+", type=str, help='order of the merging', default=["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"])

    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=6)
    input_group.add_argument('-ncw', '--nbr_CPU_worker', type=int, help='Number of worker', default=5)
    input_group.add_argument('-ngw', '--nbr_GPU_worker', type=int, help='Number of worker', default=5)


    args = parser.parse_args()
    main(args)
