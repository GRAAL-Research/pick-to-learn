import subprocess

for dataset in ['airfoil', 'concrete', 'parkinson', 'powerplant', 'infrared']:
    for model_type in ['tree', 'forest']:
        subprocess.run(['python', 'main.py', '-s', model_type, '-p', dataset])