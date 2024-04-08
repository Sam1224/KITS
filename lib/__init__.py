import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

config = {
    'logs': 'logs/'
}
datasets_path = {
    'air': 'datasets/air_quality',
    'la': 'datasets/metr_la',
    'bay': 'datasets/pems_bay',
    'sea': 'datasets/inrix_sea',
    'pems03': 'datasets/pems03',
    'pems04': 'datasets/pems04',
    'pems07': 'datasets/pems07',
    'pems08': 'datasets/pems08',
    'sea_loop': 'datasets/sea_loop',
    'nrel_al': 'datasets/nrel_al',
    'nrel_fl': 'datasets/nrel_fl',
    'nrel_ny': 'datasets/nrel_ny',
    'nrel_ma': 'datasets/nrel_ma',
    'nrel_md': 'datasets/nrel_md',
    'ushcn': 'datasets/ushcn',
    'synthetic': 'datasets/synthetic'
}
epsilon = 1e-8

for k, v in config.items():
    config[k] = os.path.join(base_dir, v)
for k, v in datasets_path.items():
    datasets_path[k] = os.path.join(base_dir, v)
