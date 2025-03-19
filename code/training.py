import argparse

parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--encoder', required=True, choices=['ResNet', 'SAM', 'effSAM']) 
args = parser.parse_args()
encoder = args.encoder

# Training script
#
# Inputs:
# --encoder
# possible options: ResNet, SAM or effSAM

# Requirements/Remarks
#
# 0. ../data folder is empty, you can download from CodeOcean
# 1. Try to set random seed for repeatability
# 2. model checkpoints should be saved to ../results/checkpoints
# 3. After the training session, you can manually copy a checkpoint 
#    in ../results/checkpoints to ../data/checkpoints if you think
#    the new model is more accurate.
# 4. If you update a checkpoint in ../data/checkpoints, make sure 
#    to update the predictions and evaluate scripts to point new checkpoints.
# 5. Training should use CUDA if it is available
# 

# Output:
# ../results/checkpoints/
# 1. There is no strict naming scheme but it would be better if 
#    it starts with {encoder} name.
print(f'Training {encoder} started')

#####
# Training goes here
#####

print(f'Training {encoder} finished')