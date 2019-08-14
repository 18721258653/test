import os
import sys
import time
import numpy as np
import cv2
import math
import PIL.Image as Image
import matplotlib.pyplot as plt
# from pysixd import view_sampler, inout, misc
# from pysixd.renderer import render
# from params.dataset_params import get_dataset_params
from os.path import join
import copy
from linemodLevelup.build import linemodLevelup_pybind
dep_max=1200
dep_min=600
dep_anchor_step=1.2
dep_anchors=[]
current_dep=dep_min
while current_dep<dep_max:
    dep_anchors.append(int(current_dep))
    current_dep=current_dep*dep_anchor_step
dep_range = 150
detector = linemodLevelup_pybind.Detector(4, [4, 8], 2)
rgb=Image.open(r'./data/2_rgb.png').convert("RGB")
# depth=Image.open(r'./data/2_PointCloud.png')
depth=cv2.imread(r'./data/2_PointCloud.png',cv2.IMREAD_GRAYSCALE)
mask = (depth <169).astype(np.uint8) * 255
# plt.figure("gray")
# plt.imshow(depth,cmap="gray")
# plt.show()
real_depth=np.zeros(depth.shape,dtype=np.uint16)
real_depth[np.where(depth <169)]=depth[np.where(depth <169)]
real_depth=np.array(real_depth)*600/255
real_depth = real_depth.astype(np.uint16)

#
# cv2.imshow("depth",mask)
# cv2.waitKey(0)
# plt.figure("rgb")
# plt.imshow(rgb)
# plt.figure("depth")
# plt.imshow((depth-600)*255/600,cmap='gray')
# plt.show()
# plt.close()
templateInfo_radius = dict()
for dep in dep_anchors:
    templateInfo_radius[dep] = dict()
success = detector.addTemplate([np.array(rgb,dtype=np.uint8), real_depth], '{:02d}_template_{}'.format(1, dep_anchors[0]),
                                               mask, dep_anchors)
template_saved_to="./data/test_%s.yaml"
tempInfo_saved_to="./data/test_{:2d}_info_{}.yaml"
print('success: {}'.format(success))
for i in range(len(dep_anchors)):
    if success[i] != -1:
        aTemplateInfo = dict()

        aTemplateInfo['cam_t_w2c'] = dep_anchors[i]

        templateInfo = templateInfo_radius[dep_anchors[i]]
        templateInfo[success[i]] = aTemplateInfo

# inout.save_info(tempInfo_saved_to.format(obj_id, radius),
#                 templateInfo_radius[radius])
detector.writeClasses(template_saved_to)
detector.clear_classes()

# into test model
template_read_classes = []
poseRefine = linemodLevelup_pybind.poseRefine()
for radius in dep_anchors:
    template_read_classes.append('{:02d}_template_{}'.format(1, radius))
detector.readClasses(template_read_classes, template_saved_to)
match_ids = list()
for radius in dep_anchors:
    match_ids.append('{:02d}_template_{}'.format(1, radius))

# srcs, score for one part, active ratio, may be too low for simple objects so too many candidates?
active_ratio = 0.6
matches = detector.match([np.array(rgb,dtype=np.uint8), real_depth], 70, active_ratio,
                         match_ids, dep_anchors, dep_range, masks=[])
print()
