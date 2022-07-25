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

#region Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

TRANSLATE ={
  "Mandible" : "MAND",
  "Maxilla" : "MAX",
  "Cranial-base" : "CB",
  "Cervical-vertebra" : "CV",
  "Root-canal" : "RC",
  "Mandibular-canal" : "MCAN",
  "Upper-airway" : "UAW",
  "Skin" : "SKIN",
  "Teeth" : "TEETH"
}

INV_TRANSLATE = {}
for k,v in TRANSLATE.items():
    INV_TRANSLATE[v] = k

LABELS = {

    "LARGE":{
        "MAND" : 1,
        "CB" : 2,
        "UAW" : 3,
        "MAX" : 4,
        "CV" : 5,
        "SKIN" : 6,
    },
    "SMALL":{
        "MAND" : 1,
        "RC" : 2,
        "MAX" : 4,
    }
}


LABEL_COLORS = {
    1: [216, 101, 79],
    2: [128, 174, 128],
    3: [0, 0, 0],
    4: [230, 220, 70],
    5: [111, 184, 210],
    6: [172, 122, 101],
}

NAMES_FROM_LABELS = {"LARGE":{}, "SMALL":{}}
for group,data in LABELS.items():
    for k,v in data.items():
        NAMES_FROM_LABELS[group][v] = INV_TRANSLATE[k]


MODELS_GROUP = {
    "LARGE": {
        "FF":
        {
            "MAND" : 1,
            "CB" : 2,
            "UAW" : 3,
            "MAX" : 4,
            "CV" : 5,
        },
        "SKIN":
        {
            "SKIN" : 1,
        }
    },


    "SMALL": {
        "HD-MAND":
        {
            "MAND" : 1
        },
        "HD-MAX":
        {
            "MAX" : 1
        },
        "RC":        
        {
            "RC" : 1
        },
    },
}

#endregion


def SaveSeg(file_path, spacing ,seg_arr, input_path,temp_path, outputdir,temp_folder, save_vtk, smoothing = 5, model_size= "LARGE"):

    print("Saving segmentation for ", file_path)

    SavePrediction(seg_arr,input_path,temp_path,output_spacing = spacing)
    # if clean_seg:
    #     CleanScan(temp_path)
    SetSpacingFromRef(
        temp_path,
        input_path,
        # "Linear",
        outpath=file_path
        )

    if save_vtk:
        SavePredToVTK(file_path,temp_folder, smoothing, out_folder=outputdir,model_size=model_size)




def CropSkin(skin_seg_arr, thickness):


    skin_img = sitk.GetImageFromArray(skin_seg_arr)
    skin_img = sitk.BinaryFillhole(skin_img)

    eroded_img = sitk.BinaryErode(skin_img, [thickness] * skin_img.GetDimension())

    skin_arr = sitk.GetArrayFromImage(skin_img)
    eroded_arr = sitk.GetArrayFromImage(eroded_img)

    croped_skin = np.where(eroded_arr==1, 0, skin_arr)

    out, N = cc3d.largest_k(
        croped_skin, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )


    return out
    
def CleanArray(seg_arr,radius):
    input_img = sitk.GetImageFromArray(seg_arr)
    output = sitk.BinaryDilate(input_img, [radius] * input_img.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [radius] * output.GetDimension())

    labels_in = sitk.GetArrayFromImage(output)
    out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )

    return out






def main(args):

    cropSize = args.crop_size

    temp_fold = os.path.join(args.temp_fold, "temp_" + id_generator())
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)



    # Find available models in folder
    available_models = {}
    print("Loading models from", args.dir_models)
    normpath = os.path.normpath("/".join([args.dir_models, '**', '']))
    for img_fn in glob.iglob(normpath, recursive=True):
        #  print(img_fn)
        basename = os.path.basename(img_fn)
        if basename.endswith(".pth"):
            model_id = basename.split("_")[1]
            available_models[model_id] = img_fn

    print("Available models:", available_models)




    # Choose models to use
    MODELS_DICT = {}
    models_to_use = {}
    # models_ID = []  
    if args.high_def:
        model_size = "SMALL"
        MODELS_DICT = MODELS_GROUP["SMALL"]
        spacing = [0.16,0.16,0.32]

    else:
        model_size = "LARGE"
        MODELS_DICT = MODELS_GROUP["LARGE"]
        spacing = [0.4,0.4,0.4]


    for model_id in MODELS_DICT.keys():
        if model_id in available_models.keys():
            for struct in args.skul_structure:
                if struct in MODELS_DICT[model_id].keys():
                    if model_id not in models_to_use.keys():
                        models_to_use[model_id] = available_models[model_id]


            # if True in [ for struct in args.skul_structure]:



    print(models_to_use)



    # load data
    data_list = []


    if args.output_dir != None:
        outputdir = args.output_dir


    number_of_scans = 0
    if os.path.isfile(args.input):  
        print("Loading scan :", args.input)
        img_fn = args.input
        basename = os.path.basename(img_fn)
        new_path = os.path.join(temp_fold,basename)
        temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
        if not os.path.exists(new_path):
            CorrectHisto(img_fn, new_path,0.01, 0.99)
        # new_path = img_fn
        data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
        number_of_scans += 1

        if args.output_dir == None:
            outputdir = os.path.dirname(args.input) 

    else:

        if args.output_dir == None:
            outputdir = args.input

        scan_dir = args.input
        print("Loading data from",scan_dir )
        normpath = os.path.normpath("/".join([scan_dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    number_of_scans += 1


        counter = 0
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    new_path = os.path.join(temp_fold,basename)
                    temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
                    if not os.path.exists(new_path):
                        CorrectHisto(img_fn, new_path,0.01, 0.99)
                    data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
                    counter += 1




    #endregion


 # region prepare data

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
    # endregion


    startTime = time.time()
    seg_not_to_clean = ["CV","RC"]


    with torch.no_grad():
        for step, batch in enumerate(pred_loader):

            #region PREDICTION

            input_img, input_path,temp_path = (batch["scan"].to(DEVICE), batch["name"],batch["temp_path"])

            image = input_path[0]
            print("Working on :",image)
            baseName = os.path.basename(image)
            scan_name= baseName.split(".")
            # print(baseName)
            pred_id = "_XXXX-Seg_"+ args.prediction_ID

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


            if args.save_in_folder:
                outputdir += "/" + scan_name[0] + "_" + "SegOut"
                print("Output dir :",outputdir)

                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                

            prediction_segmentation = {}



            for model_id,model_path in models_to_use.items():

                net = Create_UNETR(
                    input_channel = 1,
                    label_nbr= len(MODELS_DICT[model_id].keys()) + 1,
                    cropSize=cropSize
                ).to(DEVICE)


                # net = Create_SwinUNETR(
                #     input_channel = 1,
                #     label_nbr= len(MODELS_DICT[model_id].keys()) + 1,
                #     cropSize=cropSize
                # ).to(DEVICE)
                
                

                print("Loading model", model_path)
                net.load_state_dict(torch.load(model_path,map_location=DEVICE))
                net.eval()


                val_outputs = sliding_window_inference(input_img, cropSize, args.nbr_GPU_worker, net,overlap=args.precision)

                pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)

                segmentations = pred_data.permute(0,3,2,1)

                # print("Segmentations shape :",segmentations.shape)

                seg = segmentations.squeeze(0)

                seg_arr = seg.numpy()[:]



                for struct, label in MODELS_DICT[model_id].items():
                
                    sep_arr = np.where(seg_arr == label, 1,0)

                    if (struct == "SKIN"):
                        sep_arr = CropSkin(sep_arr,5)
                        # sep_arr = GenerateMask(sep_arr,20)
                    elif not True in [struct == id for id in seg_not_to_clean]:
                        sep_arr = CleanArray(sep_arr,2)

                    prediction_segmentation[struct] = sep_arr



            #endregion



            #region ===== SAVE RESULT =====

            seg_to_save = {}
            for struct in args.skul_structure:
                seg_to_save[struct] = prediction_segmentation[struct]

            save_vtk = args.gen_vtk

            if "SEPARATE" in args.merge or len(args.skul_structure) == 1:
                for struct,segmentation in seg_to_save.items():
                    file_path = os.path.join(outputdir,pred_name.replace('XXXX',struct))
                    SaveSeg(
                        file_path = file_path,
                        spacing = spacing,
                        seg_arr=segmentation,
                        input_path=input_path[0],
                        outputdir=outputdir,
                        temp_path=temp_path[0],
                        temp_folder=temp_fold,
                        save_vtk=args.gen_vtk,
                        smoothing=args.vtk_smooth,
                        model_size=model_size
                    )
                    save_vtk = False

            if "MERGE" in args.merge and len(args.skul_structure) > 1:
                print("Merging")
                file_path = os.path.join(outputdir,pred_name.replace('XXXX',"MERGED"))
                merged_seg = np.zeros(seg_arr.shape)
                for struct in args.merging_order:
                    if struct in seg_to_save.keys():
                        merged_seg = np.where(seg_to_save[struct] == 1, LABELS[model_size][struct], merged_seg)
                SaveSeg(
                    file_path = file_path,
                    spacing = spacing,
                    seg_arr=merged_seg,
                    input_path=input_path[0],
                    outputdir=outputdir,
                    temp_path=temp_path[0],
                    temp_folder=temp_fold,
                    save_vtk=save_vtk,
                    model_size=model_size
                )


            #endregion

    
    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))

    print("Done in %.2f seconds" % (time.time() - startTime))


#endregion



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #Generate mutually exclusive group
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('-id','--dir', type=str, help='Path to the scans folder', default='/app/data/scans')
    # group.add_argument('-if','--file', type=str, help='Path to the scan', default=None)
    # input_group.add_argument('--dir', type=str, help='Input directory with the scans', default='/app/data/scans')
    
    input_group = parser.add_argument_group('directory')

    input_group.add_argument('-i','--input', type=str, help='Path to the scans folder', default='/app/data/scans')
    input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models', default='/app/data/ALL_MODELS')
    # input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models', default='/app/data/ALL_MODELS')
    # input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)
    input_group.add_argument('-temp', '--temp_fold', type=str, help='temporary folder', default='..')

    input_group.add_argument('-hd','--high_def', type=bool, help='Use high def models',default=False)
    input_group.add_argument('-ss', '--skul_structure', nargs="+", type=str, help='Skul structure to segment', default=["CV","UAW","CB","MAX","MAND"])
    input_group.add_argument('-m', '--merge', nargs="+", type=str, help='merge the segmentations', default=["MERGE"])
    input_group.add_argument('-sf', '--save_in_folder', type=bool, help='Save the output in one folder', default=True)


    input_group.add_argument('-o', '--output_dir', type=str, help='Folder to save output', default=None)
    input_group.add_argument('-id', '--prediction_ID', type=str, help='Generate vtk files', default="Pred")

    input_group.add_argument('-vtk', '--gen_vtk', type=bool, help='Genrate vtk file', default=True)
    input_group.add_argument('-vtks','--vtk_smooth', type=int, help='Smoothness of the vtk', default=5)


    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.4,0.4,0.4])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    # input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[96,96,96])

    input_group.add_argument('-pr', '--precision', type=float, help='precision of the prediction', default=0.5)

    input_group.add_argument('-mo','--merging_order',nargs="+", type=str, help='order of the merging', default=["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"])

    # input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=6)
    input_group.add_argument('-ncw', '--nbr_CPU_worker', type=int, help='Number of worker', default=5)
    input_group.add_argument('-ngw', '--nbr_GPU_worker', type=int, help='Number of worker', default=1)


    args = parser.parse_args()
    main(args)
