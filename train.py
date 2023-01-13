from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.v8 import classify, detect, segment
from pprint import pprint
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT

from pathlib import Path
import os
from terminaltables import AsciiTable
from functools import partial

def log_model(trainer,core_context):
    last_weight_path = trainer.last
    print("trainer.epoch + 1: ",trainer.epoch + 1)
    print("last_weight_path: ",last_weight_path)
    pprint(trainer.label_loss_items(trainer.tloss, prefix="train"))
    pprint(trainer.label_loss_items(trainer.tloss, prefix="val"))
    pprint(trainer.metrics)
    validator = trainer.validator
    metrics = validator.metrics
    table_d=[['Class','Images','Instances','Precision','Recall','mAP50','mAP50-95']]
    mean_res = validator.metrics.mean_results()
    mean_p="{:.4f}".format(mean_res[0])
    mean_r="{:.4f}".format(mean_res[1])
    mean_mAP50="{:.4f}".format(mean_res[2])
    mean_mAP50_95= "{:.4f}".format(mean_res[3])
    table_d.append(["all",
                    validator.seen,
                    validator.nt_per_class.sum(),
                    mean_p,
                    mean_r,
                    mean_mAP50,
                    mean_mAP50_95])
    for i, c in enumerate(metrics.ap_class_index):

        table_d.append([validator.names[c],
                        validator.seen,
                        validator.nt_per_class[c],
                        "{:.4f}".format(metrics.class_result(i)[0]),
                        "{:.4f}".format(metrics.class_result(i)[1]),
                        "{:.4f}".format(metrics.class_result(i)[2]),
                        "{:.4f}".format(metrics.class_result(i)[3])]
                        )
    
    table = AsciiTable(table_d)
    print(table.table)

if __name__ == '__main__':

    FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[2]  # YOLO

    CFG = get_config(DEFAULT_CONFIG)
    experiment_params = {
                
                'save':True,
                'project':'runs', 
                'name':'test_run', 
                'exist_ok':False, 
                'verbose':False, 
                'deterministic':True, 
                'v5loader':True
                }
    augmentation_params = {
                'hsv_h':0.015, 
                'hsv_s':0.7,
                'hsv_v':0.4, 
                'degrees':0.0, 
                'translate':0.1, 
                'scale':0.5, 
                'shear':0.0, 
                'perspective':0.0, 
                'flipud':0.0, 
                'fliplr':0.5, 
                'mosaic':1.0, 
                'mixup':0.0, 
                'copy_paste':0.0,
    }
    training_params = {
                'data':"coco128.yaml",
                'pretrained':False,
                "model": 'yolov8n.pt',# test
                'epochs':1,# Number of epochs to train
                'imgsz':64, # Image size of data in dataloader
                'patience':128//5,
                'batch':16, # Batch size of the dataloader
                'cache':False, 
                'device':'None', # cuda device, i.e. 0 or 0,1,2,3 or cpu. '' selects available cuda 0 device
                'workers':8,# Number of cpu workers used per process. Scales automatically with DDP
                'optimizer':'SGD', # Optimizer used. Supported optimizer are: Adam, SGD, RMSProp
                'seed':0,
                'single_cls':False, # Train on multi-class data as single-class
                'image_weights':False, # Use weighted image selection for training
                'rect':False, # Enable rectangular training
                'cos_lr':True, #Use cosine LR scheduler
                'close_mosaic':10, 
                'resume':False, 
                'overlap_mask':True, 
                'mask_ratio':4, 
                'dropout':False,
                'lr0':0.01, # Initial learning rate
                'lrf':0.01, # Final OneCycleLR learning rate
                'momentum':0.937, # Use as momentum for SGD and beta1 for Adam
                'weight_decay':0.0005, # Optimizer weight decay
                'warmup_epochs':3.0,  # Warmup epochs. Fractions are ok.
                'warmup_momentum':0.8, # Warmup initial momentum
                'warmup_bias_lr':0.1, # Warmup initial bias lr
                'box':7.5, # Box loss gain
                'cls':0.5, # cls loss gain
                'dfl':1.5, 
                'fl_gamma':0.0, # focal loss gamma
                'label_smoothing':0.0, 
                'nbs':64, # nominal batch size
    }
    experiment_params.update(training_params)
    experiment_params.update(augmentation_params)
    print(experiment_params)

    trainer = detect.DetectionTrainer(overrides=experiment_params)


    trainer.add_callback('on_fit_epoch_end',partial(log_model,core_context='Hi'))
    trainer.train()