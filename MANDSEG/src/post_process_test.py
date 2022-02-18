from matplotlib.pyplot import axis
import cc3d
import numpy as np
import SimpleITK as sitk


input_img = sitk.ReadImage("/Users/luciacev-admin/Desktop/MANDSEG_TEST/PRED_HP/10_T0_VC_Pred.or.nii.gz") 

labels_in = sitk.GetArrayFromImage(input_img)

# labels_out = cc3d.connected_components(labels_in)
# stats = cc3d.statistics(labels_out)
# print(stats["voxel_counts"])

labels_out, N = cc3d.largest_k(
  labels_in, k=1, 
  connectivity=26, delta=0,
  return_N=True,
)

# labels_out = cc3d.dust(
#   labels_in, threshold=10, 
#   connectivity=26, in_place=False
# )
# labels_out = cc3d.dust(labels_in)

output = sitk.GetImageFromArray(labels_out)
output.SetSpacing(input_img.GetSpacing())
output.SetDirection(input_img.GetDirection())
output.SetOrigin(input_img.GetOrigin())

writer = sitk.ImageFileWriter()
writer.SetFileName("/Users/luciacev-admin/Desktop/MANDSEG_TEST/PRED_HP/10_T0_VC_Pred.or.nii.gz")
writer.Execute(output)

# closing_radius = 8
# output = sitk.BinaryDilate(output, [closing_radius] * output.GetDimension())
# output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

# writer = sitk.ImageFileWriter()
# writer.SetFileName("closed.nii.gz")
# writer.Execute(output)

# closed = sitk.GetArrayFromImage(output)

# stats = cc3d.statistics(labels_out)
# # print(stats)
# # print("mid = ", np.mean(stats['centroids'], axis = 0))
# mand_bbox = stats['bounding_boxes'][1]
# # print(mand_bbox)
# rng_lst = []
# mid_lst = []
# for slices in mand_bbox:
#   rng = slices.stop-slices.start
#   mid = (2/3)*rng+slices.start
#   rng_lst.append(rng)
#   mid_lst.append(mid)

# print(rng_lst,mid_lst)



# dif = closed - labels_out

# print(np.shape(labels_out[:,:,:150]))
# print(labels_out[:,:,:150])
# print(np.shape(closed[:,:,150:]))
# print(closed[:,:,150:])
# merge_slice = int(mid_lst[0])
# print(merge_slice)
# out = np.concatenate((labels_out[:merge_slice,:,:],closed[merge_slice:,:,:]),axis=0)


# output = sitk.GetImageFromArray(out)
# output.SetSpacing(input_img.GetSpacing())
# output.SetDirection(input_img.GetDirection())
# output.SetOrigin(input_img.GetOrigin())

# writer = sitk.ImageFileWriter()
# writer.SetFileName("/Users/luciacev-admin/Desktop/MANDSEG_TEST/PRED_HP/MARILIA-30_Pred_Sp05.nii.gz")
# writer.Execute(output)


"""

labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = cc3d.connected_components(labels_in) # 26-connected

connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

# If you're working with continuously valued images like microscopy
# images you can use cc3d to perform a very rough segmentation. 
# If delta = 0, standard high speed processing. If delta > 0, then
# neighbor voxel values <= delta are considered the same component.
# The algorithm can be 2-10x slower though. Zero is considered
# background and will not join to any other voxel.
labels_out = cc3d.connected_components(labels_in, delta=10)

# You can extract the number of labels (which is also the maximum 
# label value) like so:
labels_out, N = cc3d.connected_components(labels_in, return_N=True) # free
# -- OR -- 
labels_out = cc3d.connected_components(labels_in) 
N = np.max(labels_out) # costs a full read

# You can extract individual components using numpy operators
# This approach is slow, but makes a mutable copy.
for segid in range(1, N+1):
  extracted_image = labels_out * (labels_out == segid)
  process(extracted_image) # stand in for whatever you'd like to do

# If a read-only image is ok, this approach is MUCH faster
# if the image has many contiguous regions. A random image 
# can be slower. binary=True yields binary images instead
# of numbered images.
for label, image in cc3d.each(labels_out, binary=False, in_place=True):
  process(image) # stand in for whatever you'd like to do

# Image statistics like voxel counts, bounding boxes, and centroids.
stats = cc3d.statistics(labels_out)

# Remove dust from the input image. Removes objects with
# fewer than `threshold` voxels.
labels_out = cc3d.dust(
  labels_in, threshold=100, 
  connectivity=26, in_place=False
)

# Get a labeling of the k largest objects in the image.
# The output will be relabeled from 1 to N.
labels_out, N = cc3d.largest_k(
  labels_in, k=10, 
  connectivity=26, delta=0,
  return_N=True,
)
labels_in *= (labels_out > 0) # to get original labels

# We also include a region adjacency graph function 
# that returns a set of undirected edges.
edges = cc3d.region_graph(labels_out, connectivity=connectivity) 

# You can also generate a voxel connectivty graph that encodes
# which directions are passable from a given voxel as a bitfield.
# This could also be seen as a method of eroding voxels fractionally
# based on their label adjacencies.
# See help(cc3d.voxel_connectivity_graph) for details.
graph = cc3d.voxel_connectivity_graph(labels, connectivity=connectivity)

"""