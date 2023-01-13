# YoloV8 with DeterminedAI
Proof of Concept on how to integrate YoloV8(https://github.com/ultralytics/ultralytics) with DeterminedAI
Author: Andrew Mendez
# Installation Steps
`bash install-dep.sh`
`det deploy local cluster-up --no-gpu --master-port 8081`

# Run Training if using M1 Mac (Arm64)
`bash run_exp_arm64.sh`

# Run Training if using Linux/Mac Intel-based (Amd64)
`bash run_exp.sh`

# Steps to run Vanilla Training without Determined
`bash install-dep.sh`
`python train.py`

Any Questions, please reach out to me at andrew.mendez@hpe.com