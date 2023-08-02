
from util.Preppers import HotnCode as Prep

path = {'map':'csv/original/Train.csv',
        'val':'csv/original/Test.csv'}


Prep(path['map'], path['val'])