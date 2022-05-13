#!/usr/bin/env python
import os
import vtk
import argparse

def convertFile(filepath, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if os.path.isfile(filepath):
        basename = os.path.basename(filepath)
        print("Copying file:", basename)
        basename = os.path.splitext(basename)[0]
        outfile = os.path.join(outdir, basename+".stl")
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(filepath)
        reader.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(reader.GetOutputPort())
        writer.SetFileName(outfile)
        return writer.Write()==1
    return False

def convertFiles(indir, outdir):
    files = os.listdir(indir)
    files = [ os.path.join(indir,f) for f in files if f.endswith('.vtk') ]
    ret = 0
    print("In:", indir)
    print("Out:", outdir)
    for f in files:
        ret += convertFile(f, outdir)
    print("Successfully converted %d out of %d files." % (ret, len(files)))

def run(args):
    convertFiles(args.indir, args.outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VTK to STL converter")
    parser.add_argument('indir', help="Path to input directory.")
    parser.add_argument('--outdir', '-o', default='output', help="Path to output directory.")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    ret = args.func(args)