from ultralytics.yolo.utils import DEFAULT_CFG_PATH
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.v8 import classify, detect, segment
from pprint import pprint

from pathlib import Path
import os
from terminaltables import AsciiTable
import determined as det
from functools import partial
global epoch_last
import shutil

import torch
from torch.profiler import profile, ProfilerActivity

    
def on_train_batch_end(trainer, profiler):
    profiler.step()

 # this is to track when the final_eval is done, on_fit_epoch_end runs twice at the last epoch
def log_model(trainer,core_context):
    last_weight_path = trainer.last
    curr_epoch = int(trainer.epoch)
    print(curr_epoch, core_context.epoch_last,curr_epoch == core_context.epoch_last)
    if curr_epoch == core_context.epoch_last:
        return
    core_context.epoch_last=curr_epoch
    # print("trainer.epoch + 1: ",curr_epoch + 1)
    print("last_weight_path: ",last_weight_path)
    # pprint(trainer.label_loss_items(trainer.tloss, prefix="train"))
    losses = trainer.label_loss_items(trainer.tloss, prefix="train")
    losses.update(trainer.label_loss_items(trainer.tloss, prefix="val"))
    # pprint(trainer.label_loss_items(trainer.tloss, prefix="val"))
    print(losses)
    core_context.train.report_training_metrics(
                steps_completed=curr_epoch, metrics=losses
            )
    pprint(trainer.metrics)
    validator = trainer.validator
    metrics = validator.metrics
    table_d=[['Class','Images','Instances','Precision','Recall','mAP50','mAP50-95']]
    mean_res = validator.metrics.mean_results()
    mean_p=float("{:.4f}".format(mean_res[0]))
    mean_r=float("{:.4f}".format(mean_res[1]))
    mAP50=float("{:.4f}".format(mean_res[2]))
    mAP50_95= float("{:.4f}".format(mean_res[3]))
    table_d.append(["all",
                    validator.seen,
                    validator.nt_per_class.sum(),
                    mean_p,
                    mean_r,
                    mAP50,
                    mAP50_95])
    val_dict = {}
    val_dict["Precision"] = mean_p
    val_dict["Recall"] = mean_r
    val_dict['mAP50'] = mAP50
    val_dict['mAP50_95'] = mAP50_95
    for i, c in enumerate(metrics.ap_class_index):

        table_d.append([validator.names[c],
                        validator.seen,
                        validator.nt_per_class[c],
                        "{:.4f}".format(metrics.class_result(i)[0]),
                        "{:.4f}".format(metrics.class_result(i)[1]),
                        "{:.4f}".format(metrics.class_result(i)[2]),
                        "{:.4f}".format(metrics.class_result(i)[3])]
                        )
        # val_dict["val_"+validator.names[c]+"_P"] = metrics.class_result(i)[0]
        # val_dict["val_"+validator.names[c]+"_R"] = metrics.class_result(i)[1]
        val_dict["val_"+validator.names[c]+"_mAP50"] = metrics.class_result(i)[2]
        val_dict["val_"+validator.names[c]+"_mAP50-95"] = metrics.class_result(i)[3]
    pprint(val_dict)
    core_context.train.report_validation_metrics(steps_completed=curr_epoch, metrics=val_dict)
    table = AsciiTable(table_d)
    print(table.table)
    
    # NEW: Save checkpoint.
    # NEW: Here we are saving multiple files to our checkpoint
    # directory. 1) a model state file and 2) a file includes
    # information about the training loop state.
    print("Saving Checkpoint...")
    with core_context.checkpoint.store_path({"steps_completed": curr_epoch}) as (path, storage_id):
        print("path: ",path)
        print("trainer.save_dir: ",trainer.save_dir)
        shutil.copytree(trainer.save_dir,path / 'ckpt' )
        # torch.save(model.state_dict(), path / "checkpoint.pt")
    print("Done!")
        # with path.joinpath("state").open("w") as f:
        #     f.write(f"{epochs_completed},{core_context.info.trial.trial_id}")

def run_train(core_context,hparams):
    FILE = Path(__file__).resolve()

    CFG = get_cfg(DEFAULT_CFG_PATH)
    experiment_params = {
                
                'save':True,
                'project':hparams['project'], 
                'name':hparams['name'], 
                'exist_ok':hparams['exist_ok'], 
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
                'data':hparams['data'],
                'pretrained':hparams['pretrained'],
                "model": hparams["model"],# test
                'epochs':hparams['epochs'],# Number of epochs to train
                'imgsz':hparams['imgsz'], # Image size of data in dataloader
                'patience': hparams['patience'],
                'batch':hparams['batch'], # Batch size of the dataloader
                'cache':hparams['cache'], 
                'device':hparams['device'], # cuda device, i.e. 0 or 0,1,2,3 or cpu. '' selects available cuda 0 device
                'workers':hparams['workers'],# Number of cpu workers used per process. Scales automatically with DDP
                'optimizer':hparams['optimizer'], # Optimizer used. Supported optimizer are: Adam, SGD, RMSProp
                'seed':hparams['seed'],
                'single_cls':hparams['single_cls'], # Train on multi-class data as single-class
                # 'image_weights':hparams['image_weights'], # Use weighted image selection for training
                'rect':hparams['rect'], # Enable rectangular training
                'cos_lr':hparams['cos_lr'], #Use cosine LR scheduler
                'close_mosaic':hparams['close_mosaic'], 
                'resume': hparams['resume'], 
                'overlap_mask': hparams['overlap_mask'], 
                'mask_ratio': hparams['mask_ratio'], 
                'lr0': hparams['lr0'], # Initial learning rate
                'lrf': hparams['lrf'], # Final OneCycleLR learning rate
                'momentum': hparams['momentum'], # Use as momentum for SGD and beta1 for Adam
                'weight_decay': hparams['weight_decay'], # Optimizer weight decay
                'warmup_epochs': hparams['warmup_epochs'],  # Warmup epochs. Fractions are ok.
                'warmup_momentum': hparams['warmup_momentum'], # Warmup initial momentum
                'warmup_bias_lr': hparams['warmup_bias_lr'], # Warmup initial bias lr
                'box': hparams['box'], # Box loss gain
                'cls': hparams['cls'], # cls loss gain
                'dfl': hparams['dfl'], 
                # 'fl_gamma': hparams['fl_gamma'], # focal loss gamma
                'label_smoothing': hparams['label_smoothing'], 
                'nbs': hparams['nbs'], # nominal batch size
    }
    experiment_params.update(training_params)
    experiment_params.update(augmentation_params)
    print(experiment_params)
    core_context.epoch_last=-1
    trainer = detect.DetectionTrainer(overrides=experiment_params)
    trainer.add_callback('on_fit_epoch_end',partial(log_model,core_context=core_context))
    with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(core_context.train.get_tensorboard_path())),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
        trainer.add_callback('on_train_batch_end', partial(on_train_batch_end, profiler=prof))
        trainer.train()
    
def main(info):

    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id
    print("info.latest_checkpoint: ",info.latest_checkpoint)
    print("trial_id: ",trial_id)
    
    hparams = info.trial.hparams
    print("hparams")
    print(hparams)
    distributed=None
    
    with det.core.init(distributed=distributed) as core_context:
        run_train(core_context,hparams)
    


if __name__ == '__main__':
    info = det.get_cluster_info()

    if info is not None:
        main(info)