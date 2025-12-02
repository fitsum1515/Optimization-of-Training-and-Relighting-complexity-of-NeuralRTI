#working code with light direction (49,2)
########################################################################################################################################
import glob
import numpy as np
import cv2 as cv
from patchify import patchify
import random

class MLIC():
    def __init__(self, data_path, ld_file, light_dimension=2, src_img_type='jpg', test=False, subset=None, crop=None, patch_size=None, overlap=0):
        self.ld_file = ld_file          # .lp file name containing light directions
        self.data_path = data_path      # path to data (where light directions and images are)
        self.test = test                # whether test or train is performing
        self.patch_size = patch_size    # size of the patches 
        self.overlap = overlap          # overlap between patches 

        # get the global file name for each image in the dataset
        self.filenames = sorted(glob.glob(data_path + '/*.' + src_img_type))

        # if subset is used, reduce the number of images
        if subset:
            self.filenames = self.filenames[0:subset]
        self.num_samples = len(self.filenames)

        # get resolution of images
        self.resolution = cv.imread(self.filenames[0], 0).shape
        if crop:
            self.resolution = crop[0:2]
        self.h, self.w = self.resolution

        # -----------------------------------------------------------------------------------

        # store images into the samples variable
        self.samples = np.zeros((self.h*self.w , self.num_samples, 3), dtype=np.float32)
        for i, imPath in enumerate(self.filenames):
            if crop:
                img = cv.imread(imPath, -1)[crop[2]:crop[2]+crop[0], crop[3]:crop[3]+crop[1]]
            else:
                img = cv.imread(imPath, -1)
            
            img = np.reshape(img, (self.h*self.w, 3))
            self.samples[:,i,:] = img

        # -----------------------------------------------------------------------------------

        # read light directions from ld_file
        with open(self.ld_file) as f:
            data = f.read()
        data = data.split('\n')
        data = data[1:int(data[0])+1]  # keep the lines with light directions, remove the first one which is the number of samples
        if subset:
            data = data[0:subset]
                
        self.num_lights = len(data)

        # check that number of images and number of light directions match (only for training)
        if not self.test:
            assert self.num_samples == self.num_lights, "Number of train samples and train lights must be equal, check whether you're using the correct type of source images (default is .jpg)"

        # store light directions into the ld variable
        self.ld = np.zeros((self.num_lights, light_dimension), np.float32)
        for i, dirs in enumerate(data):
            if (len(dirs.split(' ')) == 4):
                sep = ' '
            else:
                sep = '\t'
            s = dirs.split(sep)
            if len(s) == 4:
                self.ld[i] = [float(s[l]) for l in range(1, light_dimension+1)]
            else:
                self.ld[i] = [float(s[l]) for l in range(light_dimension)]

        # -----------------------------------------------------------------------------------

        # define img_norm to normalize pixel values between 0 and 1
        if cv.imread(self.filenames[0], -1).dtype == 'uint8':
            self.img_norm = 2**8
        else:
            self.img_norm = 2**16

        # normalization
        self.samples /= self.img_norm

        # -----------------------------------------------------------------------------------
        # Apply patchify if patch_size is defined
        if self.patch_size:
            self.samples_patchified, self.ld_patchified = self.apply_patchify(self.samples, self.ld)
            print('The shape of the samples_patchified =', self.samples_patchified.shape)
            self.samples_patchified = np.transpose(self.samples_patchified, (1, 0, 2))  # Transpose for correct dimensions
            print('The shape of the samples_patchified (transposed) =', self.samples_patchified.shape)
            print('The shape of the ld_patchified =', self.ld_patchified.shape)

    def apply_patchify(self, samples, light_directions):
        num_patches_per_image = 25  # 25 patches per image (5x5 grid)
        random_pixels_per_patch = 1.0 # 10% of pixels in each patch

        # Calculate the total number of pixels we will extract (10% of the patch size)
        patch_height = self.patch_size
        patch_width = self.patch_size
        num_pixels = patch_height * patch_width  # Total pixels in one patch
        num_random_pixels = int(num_pixels * random_pixels_per_patch)  # 10% of pixels

        # Initialize list to hold random pixels for each image
        all_random_pixels_for_images = []

        # Loop through each image
        for i in range(samples.shape[1]):  # Iterate over each image
            # Reshape image to its original shape
            img = np.reshape(samples[:, i, :], (self.h, self.w, 3))

            # Apply patchify to break the image into patches
            patches = patchify(img, (self.patch_size, self.patch_size, 3), step=self.patch_size)
            patches = patches.reshape(-1, self.patch_size, self.patch_size, 3)  # Flatten patches

            random_pixels_for_image = []  # Store the random pixels for this image

            # Loop through each patch and extract random pixels
            for patch_idx in range(num_patches_per_image):
                # Flatten the patch to a 2D array of pixels (each pixel has 3 channels)
                patch = patches[patch_idx]
                flat_patch = patch.reshape(-1, 3)

                # Generate random pixel positions (same for corresponding patches across all images)
                random_indices = random.sample(range(flat_patch.shape[0]), num_random_pixels)

                # Extract the random pixels for the current patch using the random indices
                random_pixels = flat_patch[random_indices]

                # Append the random pixels for this patch to the list
                random_pixels_for_image.append(random_pixels)

            # After processing all patches, store the random pixels for this image
            all_random_pixels_for_images.append(np.concatenate(random_pixels_for_image, axis=0))

        # Now, all_random_pixels_for_images will be a list of shape (49, 20070, 3) for 10% percent
        # Convert it to a numpy array and transpose to the required shape (49, 20070, 3)
        all_random_pixels_for_images = np.array(all_random_pixels_for_images)  # Shape: (49, 20070, 3)

        
        # The light directions will be repeated for each random pixel in the image 
        all_light_directions = np.array(light_directions)  

        return all_random_pixels_for_images, all_light_directions

    def reshape_samples(self, shape):
        self.samples = np.reshape(self.samples, shape)


if __name__ == '__main__':
   
    '''
     mlic = MLIC(data_path='C:\\Users\\Fitsum\\Desktop\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material1\\Dome',
                ld_file='C:\\Users\\Fitsum\\Desktop\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material1\\Dome\\dirs.lp', 
                patch_size=64, overlap=16)'''
    pass
