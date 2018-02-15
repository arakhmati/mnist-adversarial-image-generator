import matplotlib.pyplot as plt

def plot(images, deltas, adv_images, classes=None, adv_classes=None, cmap='gray'):
    
    def reshape(x):
        return x.reshape(-1, 28, 28)
    
    images = reshape(images)
    deltas = reshape(deltas)
    adv_images = reshape(adv_images)
    
    fig = plt.figure(0)
    n_rows, h, w = images.shape
    n_cols = 3
    for i in range(n_rows):
        
        def plot_subplot(col, arr, title=''):
            ax = plt.subplot2grid((n_rows, n_cols), (i, col))
            ax.axis('off')
            ax.text(40, 20, title)
            plt.imshow(arr[i].reshape((h, w)), cmap=cmap)
        
        plot_subplot(0, images, 'Label = {}'.format(classes[i]))        
        plot_subplot(1, deltas)
        plot_subplot(2, adv_images, 'Label = {}'.format(adv_classes[i]))
        
    plt.show()
    return fig