# import of standard moduels
import tensorflow as tf
import numpy as np
import os
import gc
import json
import time
import cv2 as cv
# import matplotlib.pyplot as plt
import math

# import of custom modules
from dataset import MLIC
from network import relighting_model
import utils

# training function called for each dataset
'''
Training function that receives a MLIC dataset and build a neural RTI model.
The function will automatically save the training output inside "./models/model_name".

    Parameters
    ----------
    data_path : string
        Path to the directory containing the MLIC images
    ld_file : string
        Path to the file containig light directions
    model_name : string
        Name of the current model. A folder is created with this name containing
        the trainig outputs
    src_img_type: string
        Extension of the images in the MLIC (e.g., jpg, png, tif, etc.)
    comp_coeff: int
        Number of computed coefficients, per pixel, in the latent space

    Returns
    -------
    
'''
def training(data_path, ld_file, model_name, src_img_type, comp_coeff,patch_size,overlap):

    # create the path to save training result
    # inside model_path are stored .npy files and .hdf5 model
    # inside model_path/neural the latent space as .png or .jpg and the .json file for openlime are saved
    # inside model_path/logs are stored some plots of the training losses, if enabled

    model_path = './models/' + model_name

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(model_path+'/neural')
        os.makedirs(model_path+'/logs')

    # -----------------------------------------------------------------------------------

    # subset: use a lower number of training samples
    # crop: use a cropped version of training samples [h, w, y0, x0]

    # for the crop, notice that pixel (0,0) is on top left
    # these two parameters are useful for debug, to run the training on a small
    # amount of data and check that everything is working

    subset, crop = None, None
    #subset = 2
    # crop = [10,10,0,0]

    # -----------------------------------------------------------------------------------

    # mlic: create a MLIC object, containing all information about the dataset
    # samples: ndarray containing all the images concateneted together, shape (h*w, nun_samples, 3)
    # lights: ndarray containing light directions, shape(num_samples, light_dimension)
    # h: height of the source images
    # w: width of the source images
    # num_samples: number of source images

    mlic = MLIC(data_path='C:\\Users\\Fitsum\\Desktop\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Dome', 
                ld_file='C:\\Users\\Fitsum\\Desktop\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Dome\\dirs.lp', 
                src_img_type=src_img_type, subset=subset, crop=crop, patch_size = 64, overlap =0)
    h, w = mlic.resolution
    #num_samples = mlic.samples_patchified.shape[1]
    
    num_samples = mlic.num_samples
    print("the number of samples =",num_samples)
    num_patches = mlic.num_samples
    print("the number of num_patches =",num_patches)
    # -----------------------------------------------------------------------------------

    # some tensorflow configuration
    # this configuration is kept from the first version of the network
    # it can be changed with something else if it leads to improvements

    utils.tf_config()

    # -----------------------------------------------------------------------------------

    # autoencoder: create the model object

    # encoder_parameters and decoder_parameters are list of integers. Adding more numbers
    # to these lists will add more layers to the network. Each number corresponds to the
    # number of parameters for that layer. For example, encoder_parameters = [150,150,150]
    # will create an encoder with 3 inner-layers, each one of dimension 150.
    # keep in mind that both encoder and decoder have an additional layer by default,
    # which is the output one. For the encoder it has dimension=comp_coeff, for the decoder
    # it has dimension=num_outputs (usually 3)

    autoencoder = relighting_model(num_inputs=3*num_samples,
                                   encoder_parameters=[150,150,150],
                                   decoder_parameters=[50,50],
                                   comp_coeff=comp_coeff)

    # set hyperparameters

    # num_epochs: maximum number of training epochs
    # bl: best loss
    # be: best epoch
    # error: percentage of improvement with respect to the best epoch
    # counter: it is used to stop the training early when it stops improving
    # lr: learning rate at the beginning
    # lr_patience: number of epochs without improvement to wait before decreasing lr
    # lr_min: minimum lr. lr is not decresaed below this value
    # es_patience: number of epochs without improvement to wait before stopping the training (early stop)

    num_epochs = 50
    bl = -1
    be = -1
    error = 0.05
    counter = 0
    lr = 1e-2
    lr_patience = 3
    lr_min = 1e-10
    es_patience = 8

    # -----------------------------------------------------------------------------------

    # set of variables to keep track of losses, so the training performance

    train_loss = []
    val_loss = []
    best_loss = []
    best_epoch = []

    # -----------------------------------------------------------------------------------

    # this script is using probably a non-optimal way to read input data
    # it is implemented this way to avoid allocating too much data.
    # keep in mind that one data, in this case, has to be seen as one pixel,
    # in all the light conditions, concatented together (not the whole image).
    # limiting the amount of data this way is helpful when using high resolution images.

    # limit: a number that is used to provide input data gradually, not all at once
    # n_subset: number of subsets in which the total data are subdivided





    #limit = mlic.samples.shape[0]
    #print('the shape of mlic.samples =',mlic.samples.shape)
    limit = mlic.samples_patchified.shape[0]
    print('the shape of mlic.samples_patchified =',mlic.samples_patchified.shape)
    
    print('the value of the limit =',limit)
    n_subset = limit*num_patches // limit
    print('the value of the n_subset = ',n_subset)
    
    #n_subset = new_h*new_w*random_pixels // limit

    # -----------------------------------------------------------------------------------

    # set optimizer and loss function
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    loss = 'mean_squared_error'
    metrics = None

    # compile the model to be used
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # -----------------------------------------------------------------------------------
    #light_direction = mlic.light_directions_patchified
    light_dir = mlic.ld_patchified.shape[0]
    print('the light direction for patches =',light_dir)



    # training loop
    for epoch in range(num_epochs):
        print(f'Starting epooch {epoch+1}')

        # train uses a tricky way to avoid allocating all data at the same time
        # generate all possible indices for each pixel in each configuration in a random way

        # all_indices: array containing values from 0 to h*w*num_samples-1 in a random order
        # p_idx: pixel indices, used to access pixel positions
        # gt_idx: ground truth pixel indices, used to access the pixel that is being used as gt for the each iterations

        all_indices = np.random.choice(limit*num_samples,int(limit*num_patches),replace=False)
        p_idx = all_indices % int(limit)
        gt_idx = (all_indices // int(limit)).astype(np.uint8)

        # delete all_indices to make some room in the memory
        del all_indices
        gc.collect()

        # -----------------------------------------------------------------------------------

        # define variable to keep track of train loss and validation loss

        # tl: train loss
        # vl: val loss

        tl, vl = 0, 0

        # -----------------------------------------------------------------------------------

        # access all pixels (i.e. input data) one subset at a time (total of n_subset)
        for i in range(int(n_subset)):

            # A: subset of pixel positions, defined by limit
            # B: subset of gt pixel positions, defined by limit
            # s: subset of sample pixels
            # l: subset of light directions
            # g: subset of gt pixels
           



            A = p_idx[limit*i:limit*(i+1)] if (i < n_subset-1) else p_idx[limit*i:].astype(int)
            B = gt_idx[limit*i:limit*(i+1)] if (i < n_subset-1) else gt_idx[limit*i:]
            #s = np.reshape(mlic.samples_patchified[A], (len(A)),num_patches*3)
            s = np.reshape(mlic.samples[A], (len(A),num_samples*3))
            # print('the value of A',len(A))
            # print('the value of B', len(B))
            # print('the shape of the s =',s.shape)

            l = mlic.ld[B] #for the whole light (49,2)
            
            #l = mlic.ld_patchified[B]

            #l = np.tile(mlic.ld[B], (len(A), 1))
            #l = light_direction[B]
            #g = random_pixels[A,B,:]
            g = mlic.samples[A,B,:]

           

            # delete to make room
            del A, B
            gc.collect()

            # -----------------------------------------------------------------------------------

            # train the model and get losses

            # history: variable that keeps track of the model training, used to get losses
            
            # get partial train loss and val loss, because computed on a subset.
            # at the end of this for loop tl and vl are the total train loss and val loss
            # for the current epoch
            history = autoencoder.fit([s, l], g, batch_size=64, epochs=1, verbose=2, shuffle=False,
                                    validation_split=0.1, callbacks=None)
            tl += history.history['loss'][0]
            vl += history.history['val_loss'][0]

            del s, l, g
            gc.collect()

        # -----------------------------------------------------------------------------------

        # save the model weights only at the start (first epoch) or if the improvement is good enough
        # otherwise increase the counter
        if ((bl - vl) > error*bl) or (epoch == 0):
            counter = 0
            bl = vl
            be = epoch+1
            autoencoder.save_weights(model_path+'/autoencoder_weights')
        else:
            counter += 1

        # store all losses
        train_loss.append(tl)
        val_loss.append(vl)
        best_loss.append(bl)
        best_epoch.append(be)

        # if counter is greater than the early-stop patience, stop training and restore the model's weights of the best epoch
        if counter > es_patience:
            autoencoder.load_weights(model_path+'/autoencoder_weights')
            print('Training has stopped early, best epoch model has been restored')
            break

        # if counter is greater than the learning-rate patience, decrease the learning rate if it is greater than the minimum learning rate
        if counter > lr_patience:
            lr /= 10
            if lr >= lr_min:
                optimizer.lr.assign(lr)
                print(f'Learning rate has been updated to {lr}')

    # -----------------------------------------------------------------------------------

    save_numpy = True
    save_losstrack = False

    # # plot the losses, if enabled
    # if save_losstrack:
    #     plt.figure()
    #     plt.plot(range(1,num_epochs+1), train_loss)
    #     plt.title('Train loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/train_loss.png')
    #     plt.clf()

    #     plt.figure()
    #     plt.plot(range(1,num_epochs+1), [math.log(e) for e in train_loss])
    #     plt.title('Train loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/train_loss_log.png')
    #     plt.clf()

    #     plt.figure()
    #     plt.plot(range(1,num_epochs+1), val_loss)
    #     plt.title('Validation loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/val_loss.png')
    #     plt.clf()

    #     plt.figure()
    #     plt.plot(range(1,num_epochs+1), [math.log(e) for e in val_loss])
    #     plt.title('Validation loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/val_loss_log.png')
    #     plt.clf()

    #     plt.figure()
    #     plt.plot(range(1,len(best_loss)+1), best_loss)
    #     plt.title('Best validation loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/best_loss.png')
    #     plt.clf()

    #     plt.figure()
    #     plt.plot(range(1,len(best_loss)+1), [math.log(e) for e in best_loss])
    #     plt.title('Best validation loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/best_loss_log.png')
    #     plt.clf()

    #     plt.figure()
    #     plt.plot(range(1,len(best_epoch)+1), best_epoch)
    #     plt.title('Best epoch')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.savefig(model_path + '/logs/best_epoch.png')
    #     plt.clf()

    # del samples, lights
    # gc.collect()

    # get encoded features, running the model one more time in prediction mode, using only the encoder
    mlic.reshape_samples((h*w,num_samples*3))
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    features = encoder.predict(mlic.samples)

    del mlic
    gc.collect()

    # saving max and min of features (for each channel) used for rescaling and saving features as images
    # computed coefficients are random number, but they must be rescaled between 0 and 255 to be saved as images
    max_f = [float(np.max(features[:,i])) for i in range(comp_coeff)]
    min_f = [float(np.min(features[:,i])) for i in range(comp_coeff)]
    w_list, b_list = utils.save_net_wb(decoder=decoder)
    info = {
        'height': h,
        'width': w,
        'planes': comp_coeff,
        'samples': 50,
        'format': 'jpg',
        'quality': 95,
        'type': 'neural',
        'colorspace': 'bgr',
        'max': max_f,
        'min': min_f,
        'weights': w_list,
        'biases': b_list
    }
    with open(model_path + '/neural/info.json', 'w') as f:
        json.dump(info, f)

    # save min-max as .npy and model as .hdf5, if enabled
    # not used for the final application but very handy for debug
    if save_numpy:
        decoder.save(model_path+'/decoder.hdf5')
        np.save(model_path + '/info', np.array([x for x in max_f]+[x for x in min_f]+[h,w], dtype=np.float32))

    # rescale the features using a certain number of bit (usually 8)
    bit_feat = 8
    for i in range(comp_coeff):
        features[:,i] = np.interp(features[:,i], (min_f[i],max_f[i]), (0, 2**bit_feat-1))

    # reshape features to store them as images and do it
    features = np.reshape(features, (h,w,comp_coeff))
    for j in range(comp_coeff // 3):
        # cv.imwrite(model_path + '/neural/plane'+'_'+str(j)+'.png', features[...,3*j:3*(j+1)].astype(np.uint8))
        cv.imwrite(model_path + '/neural/plane'+'_'+str(j)+'.jpg', features[...,3*j:3*(j+1)].astype(np.uint8))

# main function
def main():
    # assign an id value for the GPU and control it is available
    device_id = 3
    num_device = 4
    #num_device = len(tf.config.list_physical_devices('GPU'))
    assert int(device_id) in range(num_device), f'Device id must be lower or equal to {num_device-1}'

    # parameters for training function
    data_path = 'C:\\Users\\Fitsum\\Desktop\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Dome'
    ld_file = 'C:\\Users\\Fitsum\\Desktop\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Dome\\dirs.lp'
    model_name = 'New_model50' # model_name will be the folder where the model is saved
    src_img_type = 'jpg'
    comp_coeff = 9
    patch_size = 64
    overlap = 0

    # call the training function for one dataset - should be generalized for many datasets
    t1 = time.time()
    with tf.device(f'/GPU:{device_id}'):

        training(data_path = data_path,
                ld_file = ld_file,
                model_name = model_name,
                src_img_type = src_img_type,
                comp_coeff = comp_coeff,
                patch_size = patch_size,
                overlap = overlap)

    t2 = time.time()
    print('done!')
    print(f'--- {int(t2-t1)//60//60} h {int(t2-t1)//60%60} m {int(t2-t1)%60} s ---')


if __name__=='__main__':
    main()