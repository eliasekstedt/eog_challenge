
import torch
import matplotlib.pyplot as plt

def show(image, title='', save=True, save_as='show.png'):
    #plt.imshow(image.to("cpu").permute(1, 2, 0))#, cmap='gray')
    plt.imshow(image.to("cpu").detach().permute(1, 2, 0))#, cmap='gray')
    plt.title(title)
    plt.axis('off')
    if save:
        plt.savefig(f"{save_as}")
        plt.figure()
        plt.close('all')
    else:
        plt.show()

def plot_performance(protocol, runpath):
    epochs = range(1, len(protocol.valcost) + 1)
    traincol = 'tab:blue'
    testcol = 'tab:red'
    # figure
    fig, (ax1, ax2) = plt.subplots(2,figsize=(6, 8))
    # cost
    ax1.plot(epochs, protocol.traincost, traincol, label='train')
    ax1.plot(epochs, protocol.valcost, testcol, label='val')
    #ax1.set_ylim([0, 20])
    ax1.legend()
    #ax1.set_xticks([])
    ax1.set_ylabel('Cost')
    ax2.plot(epochs, protocol.trainperformance, traincol, label='train')
    ax2.plot(epochs, protocol.valperformance, testcol, label='val')
    #ax2.set_ylim([0, 20])
    ax2.legend()
    ax2.set_ylabel('Performance')
    plt.tight_layout()
    plt.savefig(f'{runpath}performance.png')
    plt.figure()
    plt.close('all')

def plot_cmatrix(cmatrix, runpath):
    import seaborn as sns
    sns.heatmap(cmatrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{runpath}cmatrix')
    plt.figure()
    plt.close('all')

def make_tensor_assembly(im_lst):
    nr_col = 4
    nr_images = len(im_lst)
    assert nr_images%nr_col == 0

    assembly = None
    for i in range(nr_col, nr_images+1, nr_col):
        low, high = i-nr_col, i
        images_for_current_row = im_lst[low:high]
        assembly_row = None
        for image in images_for_current_row:
            if assembly_row is None:
                assembly_row = image
            else:
                assembly_row = torch.cat([assembly_row, image], dim=2)
        if assembly is None:
            assembly = assembly_row
        else:
            assembly = torch.cat([assembly, assembly_row], dim=1)
    return assembly