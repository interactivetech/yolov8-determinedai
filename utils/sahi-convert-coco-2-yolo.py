from sahi_coco import Coco, export_coco_as_yolov5
from sahi.utils.file import load_json, save_json
import os

DATA_DIR = '/home/psdcadmin/Documents/andrew/datasets'
COCO_DATASET_NAME='tiger-team-dataset-v6'


coco = Coco.from_coco_dict_or_path(coco_dict_or_path=os.path.join(DATA_DIR,COCO_DATASET_NAME,'annotations/instances_default.json'),
                                    image_dir=os.path.join(DATA_DIR,COCO_DATASET_NAME,'images'))
print(coco.categories)

'''
Andrew(10.2.23): CVAT defaults to annotation 1, need to subtract to 0 and 1
'''
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# init Coco objects by specifying coco dataset paths and image folder directories
# coco = Coco.from_coco_dict_or_path("coco.json")

# select only 3 categories; and map them to ids 1, 2 and 3
desired_name2id = {
  "vehicle": 0,
  "helicopter": 1}
coco.update_categories(desired_name2id)

# export updated/filtered COCO dataset
print(f"Saving updated ann file: {os.path.join(DATA_DIR,COCO_DATASET_NAME,'annotations/instances_default_updated.json')} ...")
save_json(coco.json, os.path.join(DATA_DIR,COCO_DATASET_NAME,'annotations/instances_default_updated.json'))
print("Done!")
coco = Coco.from_coco_dict_or_path(coco_dict_or_path=os.path.join(DATA_DIR,COCO_DATASET_NAME,'annotations/instances_default_updated.json'),
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
