import subprocess

# for dataset in ['infrared','parkinson','powerplant', 'concrete', 'airfoil']:
#     for model_type in ['tree', 'forest']:
#         subprocess.run(['python', 'main.py', '-s', model_type, '-p', dataset])

for dataset in ['wine', 'rice', 'statlog', 'breast_cancer', 'image_segmentation', 'mice_protein', 'htru2', 'image_segmentation']:
    subprocess.run(['python', 'main.py', '-s', 'mothernet_mlp', '-p', dataset])
