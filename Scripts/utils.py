import numpy as np
import tensorflow as tf
import glob
import os
import csv
import cv2 as cv
import math

'''
Function to get weights and biases of the decoder as lists

    Parameters
    ----------
    decoder: keras Sequential model
        The keras object containing the decoder of NeuralRTI

    Returns
    -------
    w_list: list
        List containing decoder's weights
    b_list: list
        List containing decoder's biases
'''
def save_net_wb(decoder):
    w_list, b_list = [], []
    for i,layer in enumerate(decoder.layers):
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]

        # weights' rows
        while weights.shape[0] % 4 != 0:
            weights = np.concatenate((weights, np.zeros((1,weights.shape[1]),'float32')), axis=0)
        
        if i != len(decoder.layers)-1:
            # weights' columns
            while weights.shape[1] % 4 != 0:
                weights = np.concatenate((weights, np.zeros((weights.shape[0],1),'float32')), axis=1)
            # biases
            while biases.shape[0] % 4 != 0:
                biases = np.concatenate((biases, np.zeros((1,),'float32')))

        w = np.reshape(weights.T,-1).tolist()
        w = [round(e,6) for e in w]
        b = biases.tolist()
        b = [round(e,6) for e in b]

        w_list.append(w)
        b_list.append(b)

    return w_list, b_list

# ----------------------------------------------------------------------------------------------------------------

'''
Function to calculate PSNR and SSIM of relighted images

    Parameters
    ----------
    gt_path: string
        Path to the directory containing source images of the MLIC
    est_path: string
        Path to the directory containing relighted images

    Returns
    -------
    psnr: list
        List containing per image PSNR
    ssim: list
        List containing per image SSIM
'''
def calc_metrics(gt_path, est_path):
    
    psnr, ssim = [], []

    for i in range(len(gt_path)):
        gt_img = tf.image.decode_image(tf.io.read_file(gt_path[i]))
        est_img = tf.image.decode_image(tf.io.read_file(est_path[i]))
        psnr.append(tf.image.psnr(gt_img, est_img, max_val=255))

        gt_img = tf.expand_dims(gt_img, axis=0)
        est_img = tf.expand_dims(est_img, axis=0)
        ssim.append(tf.image.ssim(gt_img, est_img, max_val=255))

    return psnr, ssim

# ----------------------------------------------------------------------------------------------------------------

'''
Average function

    Parameters
    ----------
    l: list
        List of numbers

    Returns
    -------
    _: float
        Average of the numbers in the list
'''
def average(l):
    return sum(l) / len(l)

# ----------------------------------------------------------------------------------------------------------------

'''
...

    Parameters
    ----------

    Returns
    -------
'''
def to_float(l):
    return [float(e) if not isinstance(e, str) else e for e in l]

# ----------------------------------------------------------------------------------------------------------------

'''
Function to configurate tensorflow. This configuration is kept from the first
version of the network. It can be changed with something else if
it leads to improvements

    Parameters
    ----------

    Returns
    -------
'''
def tf_config():
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess = tf.compat.v1.Session(config=config)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    tf.compat.v1.keras.backend.set_session(sess)

# ----------------------------------------------------------------------------------------------------------------

'''
Function to relight a set of images using an already trained model

    Parameters
    ----------
    ld_file : string
        Path to the file containig light directions
    model_path: string
        Path to the directory containing the model
    model_name : string
        Optional. Default is ''. Name of the specific folder containing the model.
        Can be used to specify a specific model in a folder with many models
    feat_img_type: string
        Extension of the images (e.g., jpg, png, tif, etc.) containing computed
        coefficients (latent space)
    comp_coeff: int
        Number of computed coefficients, per pixel, in the latent space
    light_dimension : int
        Number of light dimensions for RTI: 2 -> (x,y) | 3 -> (x,y,z)

    Returns
    -------
'''
def relight(ld_file, model_path, feat_img_type, comp_coeff, model_name = '', light_dimension = 2):

    model_path = os.path.join(model_path, model_name)
    
    # get the keras model and its info from the .npy file
    decoder = tf.keras.models.load_model(model_path+'/decoder.hdf5')
    info = np.load(model_path+'/info.npy')

    # maximums are the first section, minimums the second
    # height is second last element, width is last element
    max_f = info[0:comp_coeff]
    min_f = info[comp_coeff:2*comp_coeff]
    h = int(info[-2])
    w = int(info[-1])

    # scale the features from image format (usually 8 bit) to the original values
    bit_feat = 8
    features = np.zeros((h,w,comp_coeff), dtype=np.uint8)
    for j in range(comp_coeff // 3):
        features_img = cv.imread(model_path +'/neural/plane_'+str(j)+'.' + feat_img_type)
        features[...,3*j:3*(j+1)] = features_img

    features = features.astype(np.float32)
    features = np.reshape(features, (h*w,comp_coeff))

    for i in range(comp_coeff):
        features[:,i] = np.interp(features[:,i], (0, 2**bit_feat-1), (min_f[i],max_f[i]))

    # get the light direction
    # the relighted images, computed, will be stored in the folder defined by rel_path
    light_dirs = get_lights(ld_file, light_dimension)
    rel_path = './test'
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)
    
    for i,dir in enumerate(light_dirs):

        lights_list = np.tile(dir, reps=h*w)
        lights_list = np.reshape(lights_list, (h*w, light_dimension))

        input = tf.keras.layers.concatenate([features, lights_list], axis=-1)
        outputs = decoder.predict_on_batch(input)
        outputs = outputs.clip(min=0, max=1)
        outputs = np.reshape(outputs, (h, w, 3))
        outputs *= 255
        outputs = outputs.astype('uint8')
        # uncomment the following line to use RGB colorspace, comment it to use BGR
        # outputs = cv.cvtColor(outputs, cv.COLOR_BGR2RGB)
        cv.imwrite(rel_path + '/relighted' + str(i).zfill(2) + '.png', outputs)

# ----------------------------------------------------------------------------------------------------------------

'''
Function to get light directions from the ld_file

    Parameters
    ----------
    ld_file: string
        Path to the file containig light directions
    light_dimension: int
        Number of light dimensions for RTI: 2 -> (x,y) | 3 -> (x,y,z)

    Returns
    -------
    ld: ndarray
        Array containing light directions, shape (num_lights, light_dimension)
'''
def get_lights(ld_file, light_dimension = 2):
    with open(ld_file) as f:
        data = f.read()
    data = data.split('\n')
    data = data[1:int(data[0])+1] #keep the lines with light directions, remove the first one which is number of samples
            
    num_lights = len(data)

    ld = np.zeros((num_lights, light_dimension), np.float32)
    for i, dirs in enumerate(data):
        if (len(dirs.split(' ')) == 4):
            sep = ' '
        else:
            sep = '\t'
        #the line could contain the image name in first position, in that case don't take it
        s = dirs.split(sep)
        if len(s) == 4:
            ld[i] = [float(s[l]) for l in range(1,light_dimension+1)]
        else:
            ld[i] = [float(s[l]) for l in range(light_dimension)]

    return ld

if __name__ == '__main__':
    pass