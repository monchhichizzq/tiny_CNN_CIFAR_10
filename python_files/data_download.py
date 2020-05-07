
# import copy
import os
from subprocess import call
from tools import mkdir

print("")

# Downloading data
print("Downloading...")
if not os.path.exists("../data_download/cifar-10-python.tar.gz"):
    call(
        'wget -O "../data_download/cifar-10-python.tar.gz" "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")

# Extracting tar data
print("Extracting...")
extract_directory = os.path.abspath("../data_download/cifar-10-python")
print(extract_directory)
if not os.path.exists(extract_directory):
    mkdir(extract_directory)
    call(
    'tar zxvf "../data_download/cifar-10-python.tar.gz" -C "../data_download/cifar-10-python"',
    shell=True
    )
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")


# batches.mea.txt: restore the label names
# data_batch_i.bin: 50,000 images 32*32 and labels of CIFAR-10 dataset
# test_batch.bin: 10,000 test images
