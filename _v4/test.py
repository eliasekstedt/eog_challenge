

"""modes = ['mobv2', 'res34']
fc_versions = ['wo1', 'bn1', 'wo2', 'bn2', 'wo3', 'bn3']
for mode in modes:
    for fcv in fc_versions:
        if mode == 'mobv2' and fcv == 'wo1':
            continue
        print(mode, fcv)"""

"""
lst = [1, 2, 3, 4, 5]
a = (min(lst) + lst[len(lst)-3:])/2
print(a)


import numpy as np


sd = np.sqrt(0.572014)
val = np.random.normal(20, sd, int(1e6))
print(len([v for v in val if v > 24])/len(val))

"""
import numpy as np

tot = 22.95
var = 0.572014

opt = tot - 2*np.sqrt(var)
p = tot + 2*np.sqrt(var)
print(opt, p)


