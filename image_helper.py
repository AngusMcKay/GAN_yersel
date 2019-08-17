import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageHelper(object):
    def save_image(self, generated, epoch, directory):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated[count, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        fig.savefig("{}/{}.png".format(directory, epoch))
        plt.close()
        
    def makegif(self, directory):
        filenames = np.sort(os.listdir(directory))
        filenames = [ fnm for fnm in filenames if ".png" in fnm]
    
        with imageio.get_writer(directory + '/image.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(directory + filename)
                writer.append_data(image)
    
    def send_to_ui(self, generated, epoch, directory):
        for i in range(len(generated)):
            fig, axs = plt.subplots(1, 1)
            axs[0,0].imshow(generated[i, :,:,0], cmap='gray')
            axs[0,0].axis('off')
            fig.savefig("{}/{}-{}.png".format(directory, epoch, i))
            plt.close()

