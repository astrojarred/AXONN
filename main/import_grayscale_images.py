def import_grayscale_images(filetype, filepath):
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

    # creates the filepath
    globpath = filepath + '/*.' + filetype

    # initializes array that will contain all the images
    image_list = []

    # loops through all filenames in the folder matching the ending .filetype
    for filename in sorted(glob.glob(globpath)):
        # imports and converts grayscale image array to numpy array
        im = np.asarray(io.imread(filename, as_grey=True))
        image_list.append(im)

    return image_list
