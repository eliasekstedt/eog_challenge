
import os
import shutil

dataroot = '../../../data/eog/'

addresses = [[f"{dataroot}{class_dir}/{name}" for name in os.listdir(f"{dataroot}{class_dir}")] for class_dir in ['0', '1']]
addresses = addresses[0] + addresses[1]
jpg_addresses = [address for address in addresses if address.endswith('JPG')]
png_addresses = [address for address in addresses if address.endswith('png')]

print(len(jpg_addresses), len(png_addresses))

for address in jpg_addresses:
    name = address.split('/')[-1]
    new_address = f"{dataroot}jpg/{name}"
    shutil.move(address, new_address)