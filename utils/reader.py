def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    # Declare the file path for each file that is storing the MNIST data.
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    # Use one of the Python library to read a compressed file.
    # The parameter 'rb' means this file is allowed to be read only and 
    # to ensure that this file is opened in a binary mode.
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    # Images and labels of each of the data will be returned from this method.
    return images, labels
