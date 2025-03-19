import argparse

parser = argparse.ArgumentParser(description="Visualization")
parser.add_argument("--partition", help="train, test or val", required=True)
args = parser.parse_args()
partition = args.partition

# Input
# --partition train, val or test

# Requirements/Remarks
# 1. By using the predictions in 
#    ../results/{partition}/ResNet
#    ../results/{partition}/SAM
#    ../results/{partition}/effSAM
#    Create the Figure 2 in manuscript
# 2. In the figure, all three models will be used etc.

# Output
# ../results/{partition}_plots.pdf

print(f'Plot generation started')

# Code goes here

print(f'Plot generation finished')