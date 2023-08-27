
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

def show(image, runpath='', title=''):
    import matplotlib.pyplot as plt
    plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()

impath = 'data/train/6_repeat_2_1620_5257_3616.JPG'



model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # set the model to evaluation mode



image = Image.open(impath)  # open an image file
image = F.to_tensor(image)  # convert the image to a PyTorch tensor
image = image.unsqueeze(0)  # add an extra dimension for the batch
image = image.cuda()
model = model.cuda()
with torch.no_grad():
    labels = model(image)[0]['labels'].cuda()
    masks = model(image)[0]['masks']
    indices = torch.where(labels == 64)[0]
    masks = masks[indices]
    masks = torch.where(masks > 0.5, torch.tensor(1.0), torch.tensor(0.0))
    masks = masks.sum(dim=0)
    masks = torch.where(masks > 0, torch.tensor([1]).cuda(), torch.tensor([0]).cuda())
    masks = masks.expand_as(image)
    masks = (masks == 1)
    
    image = image*masks
    print(masks.shape)
    print(image.shape)
    #show(masks)
    show(image.squeeze(0))

