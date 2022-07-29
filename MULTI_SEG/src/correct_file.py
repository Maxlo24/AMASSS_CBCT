
import argparse
import glob
import sys
import os

from utils import(
	SetSpacing,
	CloseCBCTSeg,
	CorrectHisto,
)

def main(args):
	img_fn_array = []
	seg_fn_array = []

	outpath = os.path.normpath("/".join([args.out]))

	if args.dir:
		normpath = os.path.normpath("/".join([args.dir, '**', '']))
		for img_fn in glob.iglob(normpath, recursive=True):
			basename = os.path.basename(img_fn)
			if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
				if True in [txt in basename for txt in ["scan","Scan"]]:
					img_obj = {}
					img_obj["img"] = img_fn
					img_obj["out"] = outpath + img_fn.replace(args.dir,'')
					img_fn_array.append(img_obj)
				if True in [txt in basename for txt in ["seg","Seg"]]:
					img_obj = {}
					img_obj["img"] = img_fn
					img_obj["out"] = outpath + img_fn.replace(args.dir,'')
					seg_fn_array.append(img_obj)

	for img_obj in seg_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]

		if not os.path.exists(os.path.dirname(out)):
			os.makedirs(os.path.dirname(out))
		CloseCBCTSeg(image, image, args.radius)

	for img_obj in img_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]
		# out = img_obj["img"]
		
		if not os.path.exists(os.path.dirname(out)):
			os.makedirs(os.path.dirname(out))
		CorrectHisto(image, image,0.01, 0.99)

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory')

    input_group.add_argument('-rad', '--radius', type=int, help='Radius of the closing', default=3)
    
    args = parser.parse_args()
    
    main(args)