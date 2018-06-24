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


def main():
    images = load_images('images')
    batch_size = 8
    model = load_model(batch_size)
    results = run_model(model, images, batch_size)
    output_results(images, results)


def load_images(input_dir):
    import imutil
    # Load all images from the input folder
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    print('Found {} input files in directory {}'.format(len(filenames), input_dir))
    images = [imutil.decode_jpg(f, resize_to=(640,480)) for f in filenames]
    return images


def load_model(batch_size):
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = batch_size

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir='logs', config=config)

    # Local path to trained weights file
    COCO_MODEL_PATH = "mask_rcnn_coco.h5"
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


def run_model(model, images, batch_size):
    results = []
    padding = batch_size - (len(images) % batch_size)
    padded_images = images + [images[-1]] * padding
    batches = [padded_images[i:i + batch_size] for i in range(0, len(padded_images), batch_size)]
    for image_batch in batches:
        result_batch = model.detect(image_batch, verbose=1)
        results.extend(result_batch)
    return results[:len(images)]


def output_results(images, results, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    assert len(images) == len(results)
    for i in range(len(images)):
        r = results[i]
        image = images[i]
        plot = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    coco.class_names, r['scores'])
        output_filename = '{}/output_{:02d}.jpg'.format(output_dir, i)
        print("Saving {}".format(output_filename))
        plot.savefig(output_filename)


if __name__ == '__main__':
    main()
