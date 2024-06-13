"""
Mask R-CNN
Modified from nucleus example and https://github.com/codeperfectplus/Mask-RCNN-Implementation

"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/rupt/")

############################################################
#  Configurations
############################################################

class RuptConfig(Config):
    """Configuration for training on the rupt detection dataset."""
    # Give the configuration a recognizable name
    NAME = "rupt"

    STEPS_PER_EPOCH = 500
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + rupt
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 50

    # DATASET_PATH = ROOT_DIR + '/dataset/training_dataset/'

 
class InferenceConfig(RuptConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
############################################################
#  Dataset
############################################################
source = 'xray_dataset'
class RuptDataset(utils.Dataset):

    def load_img(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset rupt, and the class rupt
        self.add_class("rupt", 1, "rupt")

        #load datasets
        assert subset in ['train', 'valid']
        # dataset_dir = G:\DL_Final\Mask_RCNN\dataset\training_dataset
        dataset_dir = os.path.join(dataset_dir, subset)
        # dataset_dir = G:\DL_Final\Mask_RCNN\dataset\training_dataset\train(valid)
        filenames = os.listdir(dataset_dir)
        # filenames = ['0004633','0004820'......]
        jsonfiles , annotations = [], []
        for file in filenames:
            # file = '0004633'
            file_dir = os.path.join(dataset_dir, file)
            # file_dir =  G:\DL_Final\Mask_RCNN\dataset\training_dataset\train(valid)\0004633
            json_dir = os.path.join(file_dir, 'json')
            all_json_file = os.listdir(json_dir)
            # all_json_file = [0.json,1.json,.....]
            for each in all_json_file:
                f = os.path.join(json_dir, each)
                # f是json檔的詳細位址 /.../.../xxx.json
                jsonfiles.append(f)
                annotation = json.load(open(f))
                img_name = file + '_' + annotation['imagePath'] # 0004633_0.json
                img_path = os.path.join(file_dir, 'image') + '/' + os.path.splitext(os.path.basename(f))[0] + '.tif'
                annotations.append((annotation, img_name, img_path))
            
        print("In {source} {subset} dataset we have {number:d} annotation files."
            .format(source=source, subset=subset,number=len(jsonfiles)))
        print("In {source} {subset} dataset we have {number:d} valid annotations."
            .format(source=source, subset=subset,number=len(annotations)))
        
        labelslist = ['rupt']
        for anno , img_name , img_path in annotations:
            shapes = []
            classids = []
            if anno['shapes']:
                for shape in anno['shapes']:
                    label = shape['label']
                    if label == 'rupt':
                        classids.append(1)
                        shapes.append(shape['points'])       
            width = anno['imageWidth']
            height = anno['imageHeight']
            self.add_image(
                source,
                image_id=img_name,  # use file name as a unique image id
                path=img_path,
                width=width, height=height,
                shapes=shapes, classids=classids
            )

            # print("In {source} {subset} dataset we have {number:d} class item"
            # .format(source=source, subset=subset,number=len(labelslist)))

            for labelid, labelname in enumerate(labelslist):
                self.add_class(source,labelid,labelname)

    def load_mask(self,image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not the source dataset you want, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != source:
            return super(self.__class__, self).load_mask(image_id)

        # Convert shapes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])], dtype=np.uint8)
        #printsx,printsy=zip(*points)
        for idx, points in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            pointsy,pointsx = zip(*points)
            rr, cc = skimage.draw.polygon(pointsx, pointsy)
            mask[rr, cc, idx] = 1
        masks_np = mask.astype(np.bool)
        classids_np = np.array(image_info["classids"]).astype(np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return masks_np, classids_np
    
    def image_reference(self,image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == source:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(dataset_train, dataset_val, model):
    """Train the model."""
    # Training dataset.
    dataset_train.prepare()
 
    # Validation dataset
    dataset_val.prepare()
 
    # *** This training schedule is an example. Update to your needs ***
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')
 
############################################################
#  test
############################################################


def test(model, image_path = None, video_path=None, savedfile=None):
    assert image_path or video_path
 
     # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Colorful
        import matplotlib.pyplot as plt
        
        _, ax = plt.subplots()
        visualize.display_instances(image, boxes=r['rois'], masks=r['masks'], 
            class_ids = r['class_ids'],ax = ax,
            class_names='rupt',scores=None, show_mask=True, show_bbox=True)
        # Save output
        if savedfile == None:
            file_name = "test_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        else:
            file_name = savedfile
        plt.savefig(file_name)
        #skimage.io.imsave(file_name, testresult)
    print("Saved to ", file_name)
 
############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of your dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'last' or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=./logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to test and color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to test and color splash effect on')
    parser.add_argument('--classnum', required=False,
                        metavar="class number of your detect model",
                        help="Class number of your detector.")
    args = parser.parse_args()
 
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video or args.classnum, \
            "Provide --image or --video and  --classnum of your model to apply testing"
 
 
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
 
    # Configurations
    if args.command == "train":
        config = RuptConfig()
        dataset_train, dataset_val = RuptDataset(), RuptDataset()
        dataset_train.load_img(args.dataset,"train")
        dataset_val.load_img(args.dataset,"valid")
        config.NUM_CLASSES = len(dataset_train.class_info)
    elif args.command == "test":
        config = InferenceConfig()
        config.NUM_CLASSES = int(args.classnum)+1 # add backgrouond
        
    config.display()
 
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
 
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights
 
    # Load weights
    print("Loading weights ", weights_path)
    if args.command == "train":
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes if we change the backbone?
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        # Train or evaluate
        train(dataset_train, dataset_val, model)
    elif args.command == "test":
        # we test all models trained on the dataset in different stage
        print(os.getcwd())
        filenames = os.listdir(args.weights)
        for filename in filenames:
            if filename.endswith(".h5"):
                print(f"Load weights from {filename}")
                model.load_weights(os.path.join(args.weights,filename),by_name=True)
                save_name = args.image.split('\\')
                # savedfile_name = os.path.splitext(os.path.basename(args.image))[0] + ".jpg"
                savedfile_name = save_name[-3] + '_' + save_name[-1] + ".jpg"
                test(model, image_path=args.image,video_path=args.video, savedfile=savedfile_name)
    else:
        print("'{}' is not recognized.Use 'train' or 'test'".format(args.command))