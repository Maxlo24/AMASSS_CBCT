from utils import*
import argparse
import glob
import sys
import os


def main(args):

    print("Reading folder : ", args.input_dir)
    print("Selected spacings : ", args.spacing)

    patients = {}
    		

    spacing = args.spacing

    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:

            if "_scan" in img_fn:
                SetSpacing(img_fn,spacing,outpath=img_fn)
            else:
                SetSpacing(img_fn,spacing,interpolator="NearestNeighbor" ,outpath=img_fn)


                    
                    

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    # output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('-o','--out_dir', type=str, help='Output directory', required=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.4,0.4,0.4])

    args = parser.parse_args()
    
    main(args)