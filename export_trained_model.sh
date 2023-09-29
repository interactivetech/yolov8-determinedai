CKPT_PATH='/checkpoints/runs/test_run/weights'
OUT_DIR='./exported_weights'
cp $CKPT_PATH/best.pt $OUT_DIR/best.pt
# EXPORTED_CKPT_PATH='/checkpoints/runs/test_run/weights/'
yolo export model=$CKPT_PATH/best.pt format=engine device=0 imgsz=320
cp $CKPT_PATH/best.engine $OUT_DIR