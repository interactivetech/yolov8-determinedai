EXPORTED_MODEL='exported_weights/best.engine'
yolo predict model=$EXPORTED_MODEL source='/home/psdcadmin/Documents/09152008flight2tape1_5.mpg' imgsz=320 show verbose=False
