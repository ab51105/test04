import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from tqdm import tqdm
from glob import glob

# you can implement your own data augmentation functions

def get_positive_features(train_path_pos, feature_params):
    """
    FUNC: This function should return all positive training examples (faces)
        from 36x36 images in 'train_path_pos'. Each face should be converted
        into a HoG template according to 'feature_params'. For improved performances,
        try mirroring or warping the positive training examples.
    ARG:
        - train_path_pos: a string; directory that contains 36x36 images
                          of faces.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell.
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they
                          make things slower because the feature dimenionality
                          increases and more importantly the step size of the
                          classifier decreases at test time.
    RET:
        - features_pos: (N,D) ndarray; N is the number of faces and D is the
                        template dimensionality, which would be,
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
    """
    #########################################
    ##          you code here              ##
    #########################################
    # get image filename
    image_paths = glob(os.path.join(train_path_pos, '*.jpg'));
    
    # initialization
    hog_cell_size = feature_params['hog_cell_size'];
    N = len(image_paths);
    template_size = feature_params['template_size'];
    D = int(((template_size/hog_cell_size)**2)*31);
    features_pos = np.zeros((N,D));
    
    # hog
    for img_idx,img_name in enumerate(image_paths):
        hog_result = hog(imread(img_name), hog_cell_size);
        features_pos[img_idx,:] = np.reshape(hog_result,(1,D));
      
    #########################################
    ##          you code here              ##
    #########################################

    return features_pos 
