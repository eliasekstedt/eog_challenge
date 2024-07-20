
import os
import shutil
import pandas as pd
from PIL import Image
from torchvision.transforms import Resize
resize = Resize((224, 224))

def check_make_dir(dir_address):
    if not os.path.isdir(dir_address):
        os.makedirs(dir_address)

def create_source_column(filename):
    return f"train/{filename}"

def create_target_column(row):
    return f"{row['label']}/{row['filename']}".lower().split('.jpg')[0] + '.png'

"""
def get_address(target):
    return f"{dataroot}{target}".lower().split('.jpg')[0] + '.png'
"""
    

dataroot = '../../../data/eog/'
path = {
    'csv_ori_labeled':'csv/original/ori_labeled.csv',
    'csv_labeled':'csv/original/labeled.csv',
    'dir_ori_labeled':f"{dataroot}/train/",
}

class_dirs = ['0', '1']
for dir in class_dirs:
    check_make_dir(f"{dataroot}{dir}")

df = pd.read_csv(path['csv_ori_labeled'])
df['label'] = None
condition_extent_gt_0 = (df['extent'] > 0)
condition_damage_is_DR = (df['damage'] == 'DR')
condition_damage_not_DR = (df['damage'] != 'DR')

df.loc[condition_damage_is_DR, 'label'] = 1
df.loc[condition_damage_not_DR, 'label'] = 0

df['source'] = df['filename'].apply(create_source_column)
df['target'] = df.apply(create_target_column, axis=1)

###
###
"""
"""
print('in for')

for _, row in df.iterrows():
    source_address = f"{dataroot}{row['source']}"
    target_address = f"{dataroot}{row['target']}"
    #print(target_address)
    image = Image.open(source_address)
    image = resize(image)
    image.save(target_address)
print('out for')

###
###

df.rename(columns={'target':'address'}, inplace=True)
df = df.drop(columns=['source', 'filename'])
df = df[['label', 'extent', 'growth_stage', 'season', 'damage','address', 'ID']]



print(df)
df.to_csv(path['csv_labeled'], index=False)



































"""
print(df['label'].value_counts())
print(f"0:\t{len(os.listdir(f'{dataroot}0/'))}")
print(f"1:\t{len(os.listdir(f'{dataroot}1/'))}")
"""


"""
condition_A = (df['damage'] == 'DR')
condition_B = (df['extent'] == 0)
full_condition = condition_A & condition_B
print(condition_A)
print(condition_B)
print(full_condition)
print(df[full_condition])
"""


"""
print(df)
print(len(df))
print(df['damage'].unique())
"""