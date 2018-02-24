# --------- IMPORTS -----------

import numpy as np
import tables

# ---------- INPUTS -----------

filetype = 'png'                                     # image filetype
filepath = 'ALL_IMAGES'                              # filepath to folder with the images
hdf5_path = "quantum_data.hdf5"                      # name of hdf5 file to create
height, width = 1024, 1024                           # image dimensions
img_dtype = tables.UInt8Atom()                       # dtype of image, this is grayscale 255
num_pathologies = 14                                 # number of pathologies (including None)
pathologies_path = './pathologies_binary.csv'        # path to pathologies CSV

# ----------  DEFS  -----------


def image_name_list(filetype, filepath, withextension):
    """Given a filepath, exports a list with all the filenames
       with extension means """
    import glob

    print("Creating Label Array")

    globpath = filepath + '/*.' + filetype

    # exports names of images
    labels = sorted(glob.glob(globpath))
    if withextension:
        for i, name in enumerate(labels):
            labels[i] = labels[i][(len(filepath)+1):]
    else:
        for i, name in enumerate(labels):
            labels[i] = labels[i][:-(len(filetype)+1)]
            labels[i] = labels[i][(len(filepath)+1):]

    return labels


def export_grayscale_images_HDF5(filetype, filepath):
    """Imports images of a certain 'filetype' located at 'filepath'.
       The images are converted to grayscale and imported as numpy array.

       INPUTS:
       * filetype, type of images to be imported. String. e.g. 'png'
       * filepath, path to folder containing all the images. String.

       OUTPUTS:
       * image_list, 3D numpy array containing of shape
         (# of images, pixed height, pixel width), where each pixel is
         represented by a 0-255 grayscale value normalized between 0 and 1.

    """

    import numpy as np
    import glob
    from skimage import io
    from tqdm import tqdm

    print("Converting Images...")
    # creates the filepath
    globpath = filepath + '/*.' + filetype

    # calculates total number of images
    num_images = len(glob.glob(globpath))

    # loops through all filenames in the folder matching the ending .filetype
    for i, location in tqdm(enumerate(sorted(glob.glob(globpath))),
                            total=num_images):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = io.imread(location, as_grey=True)
        # add any image pre-processing here
        # save the image
        image_storage.append(img[None])

    print('Done !')


def get_pathologies(filepath):
    import pandas as pd
    print("Importing Pathologies List")
    data = pd.read_csv(filepath)
    data = data.values
    for i in data:
        pathology_storage.append(i[None])

# ----------  MAIN  -----------

# load image labels
labels = image_name_list(filetype, filepath, True)

# create file
img_dtype = tables.UInt8Atom()
quantum_data = tables.open_file(hdf5_path, mode='w')
image_storage = quantum_data.create_earray(quantum_data.root, 'images',
                                           img_dtype, shape=(0, height, width))
label_names = quantum_data.create_array(quantum_data.root,
                                        'image_labels', labels)
pathology_storage = quantum_data.create_earray(quantum_data.root,
                                               'pathologies', img_dtype,
                                               shape=(0, num_pathologies))

# fill HDF5 with pathologies
get_pathologies(pathologies_path)

# fill HDF5 file with images
export_grayscale_images_HDF5(filetype, filepath)
