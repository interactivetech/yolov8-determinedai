# YoloV8 with DeterminedAI
Proof of Concept on how to integrate YoloV8(https://github.com/ultralytics/ultralytics) with DeterminedAI

Author: Andrew Mendez
# Install Dependencies
`bash install-dep.sh`

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