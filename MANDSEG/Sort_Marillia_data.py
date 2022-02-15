import argparse
import glob
import sys
import os
import shutil
from src.utils import SetSpacing

def main(args):

    print("Reading folder : ", args.input_dir)
    print("Selected spacings : ", args.spacing)

    patients = {}
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            file_name = basename.split(".")[0]
            elements_1 = file_name.split("1")
            elements_2 = file_name.split("2")

            if len(elements_1)>1 or len(elements_2) >1:
                if len(elements_1)>1:
                    # print(elements_1[0])
                    patient = elements_1[0] + "-1"
                elif len(elements_2)>1:
                    # print(elements_2[0])
                    patient = elements_2[0] + "-2"
                
                if patient not in patients.keys():
                    patients[patient] = {}
                
                if True in [txt in basename for txt in ["scan","Scan"]]:
                    patients[patient]["scan"] = img_fn
                elif True in [txt in basename for txt in ["MX"]]:
                    patients[patient]["MX"] = img_fn
                # elif True in [txt in basename for txt in ["MX"]]:
                #     patients[patient]["MD"] = img_fn
                # elif True in [txt in basename for txt in ["TM"]]:
                #     patients[patient]["TM"] = img_fn
                # elif True in [txt in basename for txt in ["FACE"]]:
                #     patients[patient]["FACE"] = img_fn                          
                # elif True in [txt in basename for txt in ["VC"]]:
                #     patients[patient]["VC"] = img_fn

            # print(elements_dash)

            patient = ""
            # if len(elements_) != 0:
            #     if len(elements_) > 2:
            #         patient = elements_[0] + "_" + elements_[1]
            #     elif len(elements_) > 1:
            #         patient = elements_[0]
            # if len(elements_dash) >1:
            #     patient = elements_dash[0]

            # folder_name = os.path.basename(os.path.dirname(img_fn))
            # if folder_name in patient:
            #     folder_name = os.path.basename(os.path.dirname(os.path.dirname(img_fn)))
            # patient = folder_name + "-" + patient

            # # print(patient)

    
    # print(patients.keys())

    error = False
    invalid_patient = []
    for patient,data in patients.items():
        if "scan" not in data.keys():
            print("Missing scan for patient :",patient)
            error = True
            if patient not in invalid_patient:
                invalid_patient.append(patient)
        if "MX" not in data.keys():
            print("Missing MX segmentation patient :",patient)
            error = True
            if patient not in invalid_patient:
                invalid_patient.append(patient)
        
        # if "MD" not in data.keys():
        #     print("Missing MD segmentation patient :",patient)
        #     error = True
        # if "TM" not in data.keys():
        #     print("Missing TM segmentation patient :",patient)
        #     error = True
        # if "FACE" not in data.keys():
        #     print("Missing FACE segmentation patient :",patient)
        #     error = True
        # if "VC" not in data.keys():
        #     print("Missing VC segmentation patient :",patient)
        #     error = True

    # print(patients)

    # if error:
    #     print("ERROR : folder have missing/unrecognise files", file=sys.stderr)
    #     raise
        # if patient not in patients.keys():
        #     patients[patient] = {"dir": os.path.dirname(img_fn)}

        # if True in [txt in basename for txt in ["scan","Scan"]]:
        #     patients[patient]["scan"] = img_fn

        # elif True in [txt in basename for txt in ["seg","Seg"]]:
        #     patients[patient]["seg"] = img_fn
        # else:
        #     print("----> Unrecognise CBCT file found at :", img_fn)

    for ip in invalid_patient:
        del patients[ip]

    patient_dir = "MARILIA"
    N = 0
    Outpath = os.path.normpath("/".join([args.out,patient_dir]))

    if not os.path.exists(Outpath):
        os.makedirs(Outpath)

    for patient,data in patients.items():

        scan = data["scan"]
        seg = data["MX"]

        

            
        # file_basename = os.path.basename(scan)
        # file_name = file_basename.split(".")

        for sp in args.spacing:
            spacing = str(sp).replace(".","")
            scan_name = patient_dir + "-" + str(N) + "_scan_Sp"+ spacing + ".nii.gz"
            seg_name = patient_dir + "-" + str(N) + "_seg_Sp"+ spacing + ".nii.gz"

            SetSpacing(scan,[sp,sp,sp],os.path.join(Outpath,scan_name))
            SetSpacing(seg,[sp,sp,sp],"Linear",os.path.join(Outpath,seg_name))

        N += 1



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.5])

    args = parser.parse_args()
    
    main(args)