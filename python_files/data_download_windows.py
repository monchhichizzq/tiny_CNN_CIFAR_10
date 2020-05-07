import urllib.request
import os
import tarfile
from tools import mkdir

# Download
mkdir('../data_download')
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filepath = '../data_download/cifar-10-python.tar.gz'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)
else:
    print('Data file already exists.')

# Extracting
extract_directory = '../data_download/cifar-10-python'
if not os.path.exists('../data_download/cifar-10-python'):
    tfile = tarfile.open(filepath, 'r:gz')
    result = tfile.extractall('../data_download/')
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")