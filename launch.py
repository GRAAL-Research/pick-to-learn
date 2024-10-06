import subprocess

for dataset in ['infrared','parkinson','powerplant', 'concrete', 'airfoil']:
    for model_type in ['tree', 'forest']:
        subprocess.run(['python', 'main.py', '-s', model_type, '-p', dataset])