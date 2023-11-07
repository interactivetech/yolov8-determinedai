For Tiger Team

Command to deploy determined and run jupyter notebook
* cd /home/psdcadmin/Documents/andrew/yolov8-determindai
* conda activate det
* det deploy local cluster-up --auto-work-dir="/home/psdcadmin/Documents/andrew/" --master-port=8080
* det user login admin
* jupyter lab -ip=0.0.0.0 --port=8081 --NotebookApp.token='' --NotbookApp.password=''

If running models on live stream fails, you need to re-export.
* Here are the scripts to run:

* bash export_trained_model.sh
* bash export_pretrained_model.sh

To Bring down determined cluster
* det deploy local cluster-down