from determinedai/environments:cuda-11.3-pytorch-1.12-tf-2.11-gpu-2b7e2a1

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install ultralytics==8.0.104 \
  ipywidgets \
  terminaltables \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117