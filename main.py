import os
import sys
import random
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import skimage.io

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco
import pdb; pdb.set_trace()


# Directory to save logs and trained model
MODEL_DIR = 'logs'

# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = 'images'

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

import imutil

# Load all images from the input folder
INPUT_DIR = 'images'
OUTPUT_DIR = 'output'
filenames = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]
images = [imutil.decode_jpg(f, resize_to=(640,480)) for f in filenames]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run detection
results = model.detect(images, verbose=1)

# Visualize results
for f, r in results:
    plot = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                coco.class_names, r['scores'])
    output_filename = f.replace(INPUT_DIR, OUTPUT_DIR)
    print("Saving {}".format(output_filename))
    plot.savefig(output_filename)
