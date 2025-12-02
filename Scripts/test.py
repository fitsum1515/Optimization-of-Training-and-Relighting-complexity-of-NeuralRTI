import numpy as np
import tensorflow as tf
import utils
import glob
import cv2 as cv

def test(data_path, ld_file, model_path, model_name, src_img_type, feat_img_type):
  
    utils.tf_config()

    utils.relight(ld_file='E:\\New_project_320x320\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Test\\dirs.lp',
                  model_path='E:\\New_project_320x320\\New_Patchiy_patchiy\\models',
                  model_name='New_model50',
                  feat_img_type=feat_img_type,
                  comp_coeff=9,
                  light_dimension = 2)
    
    gt_path = sorted(glob.glob(data_path + '/*.' + src_img_type))
    est_path = sorted(glob.glob('./test' + '/*.' + 'png'))
    psnr, ssim = utils.calc_metrics(gt_path, est_path)

    psnr = utils.to_float([utils.average(psnr)])
    ssim = utils.to_float([utils.average(ssim)])

    print('PSNR ', psnr)
    print()
    print('SSIM ', ssim)

def main():

    device_id = 3
    data_path = 'E:\\New_project_320x320\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Test'
    ld_file = 'E:\\New_project_320x320\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Test\\dirs.lp'
    model_path = 'E:\\New_project_320x320\\New_Patchiy_patchiy\\models'
    model_name = 'New_model50'
    src_img_type = 'jpg'
    feat_img_type = 'jpg'

    with tf.device(f'/GPU:{device_id}'):
        test(data_path = 'E:\\New_project_320x320\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Test',
             ld_file = 'E:\\New_project_320x320\\SynthRTI-master\\SynthRTI-master_no_resize\\SynthRTI-master\\Multi\\Object2\\material9\\Test\\dirs.lp',
             model_path = 'E:\\New_project_320x320\\New_Patchiy_patchiy\\models',
             model_name = 'New_model50',
             src_img_type = src_img_type,
             feat_img_type = feat_img_type)

if __name__ == '__main__':
    main()