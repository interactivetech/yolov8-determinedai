from sahi_coco import Coco, export_coco_as_yolov5
from sahi.utils.file import load_json, save_json
import os

DATA_DIR = '/home/psdcadmin/Documents/andrew/datasets'
COCO_DATASET_NAME='tiger-team-dataset'


coco = Coco.from_coco_dict_or_path(coco_dict_or_path=os.path.join(DATA_DIR,COCO_DATASET_NAME,'annotations/instances_default2.json'),
                                    image_dir=os.path.join(DATA_DIR,COCO_DATASET_NAME,'images'))
print(coco.categories)

export_coco_as_yolov5(
    output_dir=os.path.join(DATA_DIR,COCO_DATASET_NAME+'-yolov5/'), 
    train_coco=coco, 
    val_coco=coco, 
    train_split_rate=0.2, 
    numpy_seed=1,
    mod_train_dir = os.path.join(DATA_DIR,COCO_DATASET_NAME+'-yolov5/'),
    mod_val_dir = os.path.join(DATA_DIR,COCO_DATASET_NAME+'-yolov5/'))