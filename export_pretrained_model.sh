CKPT_PATH='yolov8l.pt'
OUT_DIR='./exported_weights'
yolo export model=$CKPT_PATH format=engine device=0 imgsz=320
cp yolov8l.engine $OUT_DIR
