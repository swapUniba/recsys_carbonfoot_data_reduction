import os
import shutil


models = ['DMF', 'LightGCN', 'NGCF', 'DGCF', 'CKE', 'CFKG', 'KGNNLS']
splits = [2, 4, 6, 8, 10]
datasets = ['amazon_books_60core', 'movielens']

# select a dataset
data = 'movielens'

# for each model and for each data reductiom
for model in models:
    for split in splits:

        # define the name of the reduce dataset and execute recbole
        dataset_ = f'{data}_split_{split}'
        os.system(f"python exec_model.py {model} {dataset}")
        
        try:
            shutil.rmtree('log/')
            shutil.rmtree('log_tensorboard/')
        except:
            print('log folders already removed.')