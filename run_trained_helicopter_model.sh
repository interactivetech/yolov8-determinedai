sudo chmod -R a+rwx /checkpoints
CKPT_PATH='/checkpoints/runs/test_run2/weights'
OUT_DIR='./exported_weights'
# cp $CKPT_PATH/best.pt $OUT_DIR/best.pt
# EXPORTED_CKPT_PATH='/checkpoints/runs/test_run/weights/'
yolo export model=$CKPT_PATH/best.pt format=engine device=0 imgsz=320
cp $CKPT_PATH/best.engine $CKPT_PATH/best_heli.engine

cp $CKPT_PATH/best_heli.engine $OUT_DIR
EXPORTED_MODEL='exported_weights/best_heli.engine'
yolo predict model=$EXPORTED_MODEL source='rtsp://psdcadmin:PSDC!1psdc@192.168.11.30/onvif-media/media.amp?' imgsz=320 show verbose=False