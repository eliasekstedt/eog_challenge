
import matplotlib.pyplot as plt
import torch
import torchvision.models as models

def show(image, runpath='', title=''):
    #plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    plt.imshow(image.to("cpu").detach().permute(1, 2, 0))#, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(runpath+title+'.png')
    #plt.figure()
    #plt.close('all')
    plt.show()

using_custom = True
if using_custom:
    runpath = 'run/WF1_08_breakdown/08_15_56_32/'
    from WF1_classifier.parts.Networks import Net
    model = Net()
    model.load_state_dict(torch.load(runpath+'model.pth'))
    wgts = model.architecture.resnet.conv1.weight[1]
else:

    model = models.resnet18(pretrained=True)
    wgts = model.conv1.weight[1]


show(wgts)
















"""


def main():
    pass


if __name__ == '__main__':
    main()

"""



























