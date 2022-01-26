from utils import*
import argparse
import glob
import sys
import os


def main(args):

    print("Reading folder : ", args.input_dir)
    print("Selected spacings : ", args.spacing)

    patients = {}
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            folder_name = os.path.dirname(img_fn)
            out_folder = folder_name.replace(args.input_dir,args.out_dir)
            for spacing in args.spacing:
                file_element = basename.split('.')
                file_title = file_element[0]
                if True in [part in file_title for part in ['Sp','sp']]:
                    if 'Sp' in file_title: file_name_lst = file_title.split('Sp')
                    else : file_name_lst = file_title.split('sp')                    
                    last_elements = file_name_lst[-1].split('_')
                    sp = str(spacing).replace(".","-")
                    out_name = file_name_lst[0] + 'Sp' + sp 
                    for element in last_elements[1:]:
                        out_name += "_" + element

                else:
                    sp = str(spacing).replace(".","")
                    out_name = file_title + "_Sp" + sp
                
                for ext in file_element[1:]:
                    out_name += '.' + ext

                out_file = out_folder + "/" + out_name
                # print(out_file)
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                SetSpacing(img_fn,[spacing,spacing,spacing],out_file)


                    
                    

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out_dir', type=str, help='Output directory', required=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2])

    args = parser.parse_args()
    
    main(args)