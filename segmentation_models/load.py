import skimage.io as io
import re
import os
import json
import glob
path = '/home/rishav/Projects/Image-Segmentation/gtFine_trainvaltest/gtFine/train/aachen'
images = []
jsons = []
#for filename in os.listdir(path):
searchJson = os.path.join(path,'*_gtFine_polygons.json')
searchPoly = os.path.join(path,'*_gtFine_color.png')
image_files = glob.glob(searchPoly)
files = glob.glob(searchJson)

image_files.sort()
files.sort()


jsons = []
for f in files:
        with open(f) as d:
                jsons.append(json.load(d))

print(len(jsons[0]['objects']))


        