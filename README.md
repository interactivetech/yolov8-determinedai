# YoloV8 with DeterminedAI (WIP, Experimental)
Proof of Concept on how to integrate YoloV8(https://github.com/ultralytics/ultralytics) with DeterminedAI

Author: Andrew Mendez
# Install Dependencies
`bash install-dep.sh`

# Current Limitations

Currently the code only supports single node multi-gpu training. More work is needed to support multiple node.
If you want to train with multiple gpus, only edit the: `device` config setting. You should edit where you specify all the cuda devices you want to use. Examples are follows:
Train on single GPU: `device: 0`
Train on 4 GPUs: `device: 0,1,2,3`
Train on 8 GPUs: `device: 0,1,2,3,4,5,6,7`

# Spin up Determined Cluster (No GPU cluster)
`det deploy local cluster-up --no-gpu --master-port 8081`

# Spin up Determined Cluster (GPU cluster)
`det deploy local cluster-up --no-gpu --master-port 8081`

# Run Training if using M1 Mac (Arm64)
`bash run_exp_arm64.sh`

# Run Training if using Linux/Mac (Intel-based, Amd64)
`bash run_exp.sh`

# Steps to run Vanilla Training without Determined
`python train.py`

# Questions
Any Questions, please reach out to me at andrew.mendez@hpe.com
