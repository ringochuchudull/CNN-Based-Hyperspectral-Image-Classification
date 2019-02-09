import numpy as np
from random import shuffle
import scipy.io as io
import argparse
from helper import *
import threading
import time
import itertools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines', help='Default: Indian_pines, options: Salinas, KSC, Botswana')
parser.add_argument('--train_ratio', type=float, default=0.2)
parser.add_argument('--validation_ratio', type=float, default=0.05)
parser.add_argument('--channel_first', type=bool, default=False, help='Image channel located on the last dimension')
parser.add_argument('--dtype', type=str, default='float32', help='Data type (Eg float64, float32, float16, int64...')
parser.add_argument('--plot', type=bool, default=False, help='TRUE to plot satellite images and ground truth at the end')
opt = parser.parse_args()

# Try loading data from the folder... Otherwise download from online
input_mat, target_mat = maybeDownloadOrExtract(opt.data)

# Output data type
datatype = getdtype(opt.dtype)
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
OUTPUT_CLASSES = np.max(target_mat)
PATCH_SIZE = 5

# Normalize image data and select datatype
input_mat = input_mat.astype(datatype)
input_mat = input_mat - np.min(input_mat)
input_mat = input_mat / np.max(input_mat)

# Extract a list that contains the class number with sufficient training samples
list_labels = getListLabel(opt.data)

# For showing a animation only
end_loading = False
def animate():
    global end_loading
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if end_loading:
            break
        sys.stdout.write('\rExtracting '+ opt.data + ' dataset features...' + c)
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\rFinished!\t')

print("+-------------------------------------+")
print('Input_mat shape: ' + str(input_mat.shape))

MEAN_ARRAY = np.ndarray(shape=(BAND, 1))
new_input_mat = []

input_mat = np.transpose(input_mat, (2, 0, 1))

calib_val_pad = int((PATCH_SIZE - 1)/2)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[i, :, :])
    new_input_mat.append(np.pad(input_mat[i, :, :], calib_val_pad, 'constant', constant_values=0))

input_mat = np.array(new_input_mat)

def Patch(height_index, width_index):

    # Input:
    # Given the index position (x,y) of spatio dimension of the hyperspectral image,

    # Output:
    # a data cube with patch size S (24 neighbours), with label based on central pixel

    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)

    patch = input_mat[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch).astype(datatype)


# Assign empty array to store patched images
CLASSES = []
for i in range(OUTPUT_CLASSES):
    CLASSES.append([])

# Assign empty array to count samples in each class
class_label_counter = [0] * OUTPUT_CLASSES

# Start timing for loading
t = threading.Thread(target=animate).start()
start = time.time()

count = 0
for i in range(HEIGHT-1):
    for j in range(WIDTH-1):
        curr_inp = Patch(i, j)
        curr_tar = target_mat[i, j]

        if curr_tar:
            CLASSES[curr_tar-1].append(curr_inp)
            class_label_counter[curr_tar-1] += 1
            count += 1

end_loading = True
end = time.time()
print("Total excution time..." + str(end-start)+'seconds')
print('Total number of samples: ' + str(count))
showClassTable(class_label_counter)

TRAIN_PATCH, TRAIN_LABELS = [], []
TEST_PATCH, TEST_LABELS =[], []
VAL_PATCH, VAL_LABELS = [], []

train_ratio = opt.train_ratio
val_ratio = opt.validation_ratio
# test_ratio = reminder of data

counter = 0  # Represent train_index position
for i, data in enumerate(CLASSES):
    datasize = []
    if i + 1 in list_labels:

        shuffle(data)
        print('Class ' + str(i + 1) + ' is accepted')

        size = round(class_label_counter[i]*train_ratio)

        TRAIN_PATCH += data[:size]
        TRAIN_LABELS += [counter] * size
        datasize.append(size)

        size1 = round(class_label_counter[i]*val_ratio)
        VAL_PATCH += data[size:size+size1]
        VAL_LABELS += [counter] * (size1)
        datasize.append(size1)

        TEST_PATCH += data[size+size1:]
        TEST_LABELS += [counter] * len(data[size+size1:])
        datasize.append(len(TEST_PATCH))

        counter += 1

    else:
        print('-Class ' + str(i + 1) + ' is rejected due to insufficient samples')

TRAIN_LABELS = np.array(TRAIN_LABELS)
TRAIN_PATCH = np.array(TRAIN_PATCH)
TEST_PATCH = np.array(TEST_PATCH)
TEST_LABELS = np.array(TEST_LABELS)
VAL_PATCH = np.array(VAL_PATCH)
VAL_LABELS = np.array(VAL_LABELS)

print("+-------------------------------------+")
print("Size of Training data: " + str(len(TRAIN_PATCH)) )
print("Size of Validation data: " + str(len(VAL_PATCH))  )
print("Size of Testing data: " + str(len(TEST_PATCH)) )
print("+-------------------------------------+")

CHANNEL_FIRST = opt.channel_first

train_idx = list(range(len(TRAIN_PATCH)))
shuffle(train_idx)
TRAIN_PATCH = TRAIN_PATCH[train_idx]
if CHANNEL_FIRST:
    TRAIN_PATCH = np.transpose(TRAIN_PATCH, (0, 2, 3, 1))
TRAIN_LABELS = OnehotTransform(TRAIN_LABELS[train_idx])
train = {}
train["train_patch"] = TRAIN_PATCH
train["train_labels"] = TRAIN_LABELS
io.savemat("./data/" + opt.data + "_Train_patch_" + str(PATCH_SIZE) + ".mat", train)


test_idx = list(range(len(TEST_PATCH)))
shuffle(test_idx)
TEST_PATCH = TEST_PATCH[test_idx]
if CHANNEL_FIRST:
    TEST_PATCH = np.transpose(TEST_PATCH, (0, 2, 3, 1))
TEST_LABELS = OnehotTransform(TEST_LABELS[test_idx])
test = {}
test["test_patch"] = TEST_PATCH
test["test_labels"] = TEST_LABELS
io.savemat("./data/" + opt.data + "_Test_patch_" + str(PATCH_SIZE) + ".mat", test)


val_idx = list(range(len(VAL_PATCH)))
shuffle(val_idx)
VAL_PATCH = VAL_PATCH[val_idx]
if CHANNEL_FIRST:
    VAL_PATCH = np.transpose(VAL_PATCH, (0, 2, 3, 1))
VAL_LABELS = OnehotTransform(VAL_LABELS[val_idx])
val = {}
val["val_patch"] = VAL_PATCH
val["val_labels"] = VAL_LABELS
io.savemat("./data/" + opt.data + "_Val_patch_" + str(PATCH_SIZE) + ".mat", val)

print("+-------------------------------------+")
print("Summary")
print('Train_patch.shape: '+ str(TRAIN_PATCH.shape) )
print('Train_label.shape: '+ str(TRAIN_LABELS.shape) )
print('Test_patch.shape: ' + str(TEST_PATCH.shape))
print('Test_label.shape: ' + str(TEST_LABELS.shape))
print("Validation batch Shape: " + str(VAL_PATCH.shape) )
print("Validation label Shape: " + str(VAL_LABELS.shape) )
print("+-------------------------------------+")
print("\nFinished processing.......")


if opt.plot:
    print('\n Looking at some sample images')
    plot_random_spec_img(TRAIN_PATCH, TRAIN_LABELS)
    plot_random_spec_img(TEST_PATCH, TEST_LABELS)
    plot_random_spec_img(VAL_PATCH, VAL_LABELS)

    GroundTruthVisualise(target_mat)

