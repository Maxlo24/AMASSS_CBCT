import argparse
import glob
import sys
import os
import SimpleITK as sitk
import numpy as np

def main(args):

	# outpath = os.path.normpath("/".join([args.out]))

    structures_to_merge = args.structures
    structures_labels = args.labels
    label_dic = {}
    for i in range(len(structures_to_merge)):
        label_dic[structures_to_merge[i]] = structures_labels[i]
    


    patients = {}
    if args.input:
        normpath = os.path.normpath("/".join([args.input, '**', '']))
        for img_fn in glob.iglob(normpath, recursive=True):
            basename = os.path.basename(img_fn)
            if os.path.isfile(img_fn) and True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if True in [txt in basename for txt in ["Seg"]]:
                    patient_seg = basename.split("Seg")[0][:-1].split("_")
                    seg_id = patient_seg.pop(-1)
                    patient = "_".join(patient_seg)
                #     # print(patient)
                #     # print(seg_id)
                    if patient not in patients.keys():
                        patients[patient] = {}
                        patients[patient]["dir"] = os.path.dirname(img_fn)

                    patients[patient][seg_id] = img_fn

                # if "_scan" in basename:
                #     patient = basename.split("_scan")[0]
                #     print(patient)
                #     if patient not in patients.keys():
                #         patients[patient] = {}
                #     patients[patient]["scan"] = img_fn




        # seg_fn_array.append(img_obj)
    # print(patients)

    for patient,data in patients.items():
        merge_lst = []

        print(patient)

        
        for id in args.merging_order:
            if id in data.keys() and id in structures_to_merge:
                merge_lst.append(id)
                
        first_id = merge_lst.pop(0)
        first_img = sitk.ReadImage(data[first_id])
        seg = sitk.GetArrayFromImage(first_img)
        merged_seg = np.where(seg==1,label_dic[first_id],seg)

        for id in merge_lst:
            img = sitk.ReadImage(data[id])
            seg = sitk.GetArrayFromImage(img)
            merged_seg = np.where(seg==1,label_dic[id],merged_seg)



        # for i in range(len(merge_lst)-1):
        #     label = i+2
        #     img = sitk.ReadImage(merge_lst[i+1])
        #     seg = sitk.GetArrayFromImage(img)
        #     main_seg = np.where(seg==1,label,main_seg)

        output = sitk.GetImageFromArray(merged_seg)
        output.SetSpacing(first_img.GetSpacing())
        output.SetDirection(first_img.GetDirection())
        output.SetOrigin(first_img.GetOrigin())
        output = sitk.Cast(output, sitk.sitkInt16)

        writer = sitk.ImageFileWriter()
        writer.SetFileName( os.path.join(data["dir"], patient+"_MERGED_Seg.nii.gz") )
        writer.Execute(output)





if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Merge segmentations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input', type=str, help='Input directory with 3D segmentations',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', default=parser.parse_args().input)

    input_group.add_argument('-s', '--structures', type=int, help='Structures to merge', default=["MAND","CB","MAX","CV","UAW"])
    input_group.add_argument('-l', '--labels', type=int, help='Labels of each structures', default=[1,2,2,5,3])

    input_group.add_argument('-mo','--merging_order',nargs="+", type=str, help='order of the merging', default=["CV","SKIN","UAW","CB","MAX","MAND","CAN","RCL","RCU"])


    
    args = parser.parse_args()
    
    main(args)