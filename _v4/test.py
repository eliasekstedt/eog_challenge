

"""modes = ['mobv2', 'res34']
fc_versions = ['wo1', 'bn1', 'wo2', 'bn2', 'wo3', 'bn3']
for mode in modes:
    for fcv in fc_versions:
        if mode == 'mobv2' and fcv == 'wo1':
            continue
        print(mode, fcv)"""

lst = [1, 2, 3, 4, 5]
a = (min(lst) + lst[len(lst)-3:])/2
print(a)