

hparam = {'batch_size': 200,
            'nr_epochs': 15,
            'architecture_name':'res18fc',
            'weight_decay': 1e-7,
            'dropout_rate': 0.0,
            'resizes':(128, 128)}

runpath = 'run/00_00_00_00/'

sub_name_components = runpath
sub_name_components = [runpath]+[str(param)+'_' for param in hparam.values()]
print(sub_name_components)
sub_name = ''
for i in range(len(sub_name_components)):
    sub_name = sub_name + sub_name_components[i]
sub_name = sub_name[:-1] + '.csv'

print(sub_name)