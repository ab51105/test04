import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from tqdm import tqdm
from glob import glob

# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
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
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################
    # get image filename
    image_paths = glob(os.path.join(non_face_scn_path, '*.jpg'));
    
    # initilation
    hog_cell_size = feature_params['hog_cell_size'];
    N = len(image_paths);
    template_size = feature_params['template_size'];
    D = int(((template_size/hog_cell_size)**2)*31);
    features_neg = np.zeros((num_samples,D));
    neg_examples = 'TODO';
    
    img_num = np.random.choice(N-1,num_samples);
    for i in tqdm(range(num_samples)):
        image = imread(image_paths[img_num[i]]);
        image = color.rgb2grey(image);
        start_i = random.randint(0,image.shape[0]-template_size);
        start_j = random.randint(0,image.shape[1]-template_size);
        patch = image[start_i:start_i+template_size,start_j:start_j+template_size];
        
        #hog
        hog_feats = hog(patch,hog_cell_size);
        features_neg[i,:] = np.reshape(hog_feats,(1,D));

    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg, neg_examples

