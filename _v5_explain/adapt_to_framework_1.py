
import os
from PIL import Image
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

resize = Resize((224, 224))

def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def resize_images(paths):
    for path in paths:
        address = f"{dataroot}{path}"
        image = Image.open(address)
        image = resize(image)
        target_address = address.split('.JPG')[0] + '.png'
        #image.save(target_address)







dataroot = '../../../data/eog/'



class_dirs = os.listdir(dataroot)
image_addresses = [[f"{class_dir}/{name}" for name in os.listdir(f"{dataroot}{class_dir}")] for class_dir in class_dirs]
image_addresses = image_addresses[0] + image_addresses[1]
#image_addresses = [address for address in image_addresses if address.endswith('.JPG')]



resize_images(image_addresses)









