import getpass
import string
from matplotlib.pyplot import axis
import cc3d
import numpy as np
import SimpleITK as sitk
import itk
from utils import *

seg_path_dic = {
  "MAND":"/Users/luciacev-admin/Desktop/MANDSEG_TEST/CP66_MAND_Pred_sp0-25.nii.gz",
  "SKIN":"/Users/luciacev-admin/Desktop/MANDSEG_TEST/CP66_SKIN_Pred_sp0-25.nii.gz",
  "CV":"/Users/luciacev-admin/Desktop/MANDSEG_TEST/CP66_CV_Pred_sp0-25.nii.gz",
  "CB":"/Users/luciacev-admin/Desktop/MANDSEG_TEST/CP66_CB_Pred_sp0-25.nii.gz",
  "MAX":"/Users/luciacev-admin/Desktop/MANDSEG_TEST/CP66_MAX_Pred_sp0-25.nii.gz",
  }

merging_order = ["CV","CB","MAX","MAND"]

outpath = "test.nii.gz"

MergeSeg(seg_path_dic,outpath,merging_order)



# #Get image from url
# def get_image_from_url(url):
#     import urllib.request
#     import io
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = np.reshape(image, (256, 256))
#     return image



# #plot image
# def plot_image(image):
#     plt.imshow(image,cmap='gray')
#     plt.show()

# plot_image(get_image_from_url('https://www.google.com/search?q=chat&rlz=1C5GCEM_enUS964US964&sxsrf=APq-WBsq5hGK6kOKg2_s2qx7Er00jJ8_jg:1648668396144&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjQr-qwyO72AhXEXM0KHYY4DfIQ_AUoAXoECAEQAw&biw=1680&bih=882&dpr=2#imgrc=YVqXM2zc5FB_5M'))


























# # img = itk.imread("/Users/luciacev-admin/Desktop/Scans_RS/UoP/Segs/UoP-362_seg_Sp05.nii.gz")

# # img_info = itk.template(img)[1]
# # pixel_type = img_info[0]
# # pixel_dimension = img_info[1]
# # ImageType = itk.Image[pixel_type, pixel_dimension]


# # ImageType = itk.Image[itk.US, 3]
# # BinaryFillholeImageFilter = itk.BinaryFillholeImageFilter[ImageType].New()
# # BinaryFillholeImageFilter.SetInput(img)
# # BinaryFillholeImageFilter.SetForegroundValue(1)
# # BinaryFillholeImageFilter.Update()
# # filled_itk_img = BinaryFillholeImageFilter.GetOutput()

# # itk.imwrite(filled_itk_img,"test.nii.gz")

# # output = sitk.ReadImage("/Users/luciacev-admin/Desktop/Maxime segmentations 2/AH1-seg-TM.gipl.gz")
# # closing_radius = 1
# # output = sitk.BinaryDilate(output, [closing_radius] * output.GetDimension())
# # output = sitk.BinaryFillhole(output)
# # output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

# # writer = sitk.ImageFileWriter()
# # writer.SetFileName("SKIN_FILL.nii.gz")
# # writer.Execute(output)



# input_img = sitk.ReadImage("/Users/luciacev-admin/Desktop/test/RC-P1_scan_Sp05.nii.gz") 
# input_seg = sitk.ReadImage("/Users/luciacev-admin/Desktop/test/RC-P1_seg_Sp05.nii.gz") 


# closing_radius = 1

# output = sitk.BinaryDilate(input_seg, [closing_radius] * input_seg.GetDimension())
# output = sitk.BinaryFillhole(output)
# output = sitk.GetArrayFromImage(output)
# output = np.transpose(output, (2, 0, 1))
# # output, N = cc3d.largest_k(
# #   labels_in, k=1, 
# #   connectivity=26, delta=0,
# #   return_N=True,
# # )
# output = cc3d.connected_components(output)
# output = np.transpose(output, (1, 2, 0))

# # closing_radius = 3

# # output = sitk.GetImageFromArray(output)
# # output = sitk.BinaryDilate(output, [closing_radius] * output.GetDimension())
# # output = sitk.BinaryFillhole(output)
# # output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

# stats = cc3d.statistics(output)
# tooth = stats['bounding_boxes'][1]
# # print(tooth)

# output = output[tooth[0].start:tooth[0].stop,tooth[1].start:tooth[1].stop,tooth[2].start:tooth[2].stop ]

# # print(stats["voxel_counts"])


# # labels_out = cc3d.dust(
# #   labels_in, threshold=10, 
# #   connectivity=26, in_place=False
# # )
# # labels_out = cc3d.dust(labels_in)



# output = sitk.GetImageFromArray(output)
# output.SetSpacing(input_img.GetSpacing())
# output.SetDirection(input_img.GetDirection())
# output.SetOrigin(input_img.GetOrigin())

# writer = sitk.ImageFileWriter()
# writer.SetFileName("test.nii.gz")
# writer.Execute(output)

# # closing_radius = 8
# # output = sitk.BinaryDilate(output, [closing_radius] * output.GetDimension())
# # output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

# # writer = sitk.ImageFileWriter()
# # writer.SetFileName("closed.nii.gz")
# # writer.Execute(output)

# # closed = sitk.GetArrayFromImage(output)

# # stats = cc3d.statistics(labels_out)
# # # print(stats)
# # # print("mid = ", np.mean(stats['centroids'], axis = 0))
# # mand_bbox = stats['bounding_boxes'][1]
# # # print(mand_bbox)
# # rng_lst = []
# # mid_lst = []
# # for slices in mand_bbox:
# #   rng = slices.stop-slices.start
# #   mid = (2/3)*rng+slices.start
# #   rng_lst.append(rng)
# #   mid_lst.append(mid)

# # print(rng_lst,mid_lst)



# # dif = closed - labels_out

# # print(np.shape(labels_out[:,:,:150]))
# # print(labels_out[:,:,:150])
# # print(np.shape(closed[:,:,150:]))
# # print(closed[:,:,150:])

# # merge_slice = int(mid_lst[0])
# # print(merge_slice)
# # out = np.concatenate((labels_out[:merge_slice,:,:],closed[merge_slice:,:,:]),axis=0)


# # output = sitk.GetImageFromArray(out)
# # output.SetSpacing(input_img.GetSpacing())
# # output.SetDirection(input_img.GetDirection())
# # output.SetOrigin(input_img.GetOrigin())

# # writer = sitk.ImageFileWriter()
# # writer.SetFileName("/Users/luciacev-admin/Desktop/MANDSEG_TEST/PRED_HP/MARILIA-30_Pred_Sp05.nii.gz")
# # writer.Execute(output)


# """

# labels_in = np.ones((512, 512, 512), dtype=np.int32)
# labels_out = cc3d.connected_components(labels_in) # 26-connected

# connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
# labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

# # If you're working with continuously valued images like microscopy
# # images you can use cc3d to perform a very rough segmentation. 
# # If delta = 0, standard high speed processing. If delta > 0, then
# # neighbor voxel values <= delta are considered the same component.
# # The algorithm can be 2-10x slower though. Zero is considered
# # background and will not join to any other voxel.
# labels_out = cc3d.connected_components(labels_in, delta=10)

# # You can extract the number of labels (which is also the maximum 
# # label value) like so:
# labels_out, N = cc3d.connected_components(labels_in, return_N=True) # free
# # -- OR -- 
# labels_out = cc3d.connected_components(labels_in) 
# N = np.max(labels_out) # costs a full read

# # You can extract individual components using numpy operators
# # This approach is slow, but makes a mutable copy.
# for segid in range(1, N+1):
#   extracted_image = labels_out * (labels_out == segid)
#   process(extracted_image) # stand in for whatever you'd like to do

# # If a read-only image is ok, this approach is MUCH faster
# # if the image has many contiguous regions. A random image 
# # can be slower. binary=True yields binary images instead
# # of numbered images.
# for label, image in cc3d.each(labels_out, binary=False, in_place=True):
#   process(image) # stand in for whatever you'd like to do

# # Image statistics like voxel counts, bounding boxes, and centroids.
# stats = cc3d.statistics(labels_out)

# # Remove dust from the input image. Removes objects with
# # fewer than `threshold` voxels.
# labels_out = cc3d.dust(
#   labels_in, threshold=100, 
#   connectivity=26, in_place=False
# )

# # Get a labeling of the k largest objects in the image.
# # The output will be relabeled from 1 to N.
# labels_out, N = cc3d.largest_k(
#   labels_in, k=10, 
#   connectivity=26, delta=0,
#   return_N=True,
# )
# labels_in *= (labels_out > 0) # to get original labels

# # We also include a region adjacency graph function 
# # that returns a set of undirected edges.
# edges = cc3d.region_graph(labels_out, connectivity=connectivity) 

# # You can also generate a voxel connectivty graph that encodes
# # which directions are passable from a given voxel as a bitfield.
# # This could also be seen as a method of eroding voxels fractionally
# # based on their label adjacencies.
# # See help(cc3d.voxel_connectivity_graph) for details.
# graph = cc3d.voxel_connectivity_graph(labels, connectivity=connectivity)

# """