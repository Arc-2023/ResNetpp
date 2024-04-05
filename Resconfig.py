import torch

base_path = r"D:/dataset/roads/"
base_TBAD_csv_path = r"D:/dataset/med/imageTBAD/dataframe.csv"

params = {
    'epochs': 10,
    'batch_size': 1,
    'lr': 0.0003,
    'amp': False,
    'shuffle': True,
    'in_channel': 1,
    'out_channel': 3,
    'T_0': 1,
    'T_mult': 2,
    'proportion': 0.9,
    'cos': True,
    'is_parallel': False
}
# label : b 512 512
#         b 3 512 512
# [[012],
#  [012]]
# [100
#  100]
# [010
#  010]
# [001
#  001]
# img :   b 3 512 512