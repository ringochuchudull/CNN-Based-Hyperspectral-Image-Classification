
def getdtype(t):
    import numpy as np
    if t == 'float64':
        return np.float64
    elif t == 'float32':
        return np.float32
    elif t == 'float16':
        return np.float16
    elif t == 'int64':
        return np.int64
    elif t == 'int32':
        return np.int32
    elif t == 'int16':
        return np.int16
    elif t == 'int8':
        return np.int8
    else:
        # Default value
        return np.float64

#Get Dataset
def maybeExtract(data, patch_size):
    import scipy.io
    try:
        TRAIN = scipy.io.loadmat("./data/" + data + "_Train_patch_" + str(patch_size) + ".mat")
        VALIDATION = scipy.io.loadmat("./data/" + data + "_Val_patch_" + str(patch_size) + ".mat")
        TEST = scipy.io.loadmat("./data/" + data + "_Test_patch_" + str(patch_size) + ".mat")

    except:
        raise Exception('--data options are: Indian_pines, Salinas, KSC, Botswana OR data files not existed')

    return TRAIN, VALIDATION, TEST


def maybeDownloadOrExtract(data):
    import scipy.io as io
    import os
    # Somehow this is necessary, even I cannot tell why -_-
    if data in ('KSC', 'Botswana'):
        filename = data
    else:
        filename = data.lower()

    print("Dataset: " + filename)

    try:
        print("Try using images from Data folder...")
        input_mat = io.loadmat('./data/' + data + '.mat')[filename]
        target_mat = io.loadmat('./data/' + data + '_gt.mat')[filename + '_gt']

    except:
        print("Data not found, downloading input images and labelled images!\n\n")
        if data == "Indian_pines":
            url1 = "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

        elif data == "Salinas":
            url1 = "http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

        elif data == "KSC":
            url1 = "http://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat"

        elif data == "Botswana":
            url1 = "http://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat"

        else:
            raise Exception("Available datasets are:: Indian_pines, Salinas, KSC, Botswana")

        os.system('wget -P' + ' ' + './data/' + ' ' + url1)
        os.system('wget -P' + ' ' + './data/' + ' ' + url2)

        input_mat = io.loadmat('./data/' + data + '.mat')[filename]
        target_mat = io.loadmat('./data/' + data + '_gt.mat')[filename + '_gt']

    return input_mat, target_mat


def getListLabel(data):
    if data == 'Indian_pines':
        return [2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15]

    elif data == 'Salinas':
        return list(range(1,16+1))

    elif data == 'Botswana':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  13, 14]

    elif data == 'KSC':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    else:
        raise Exception("Type error")



def OnehotTransform(labels):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)

    labels = np.reshape(labels, (len(labels), 1))
    labels = onehot_encoder.fit_transform(labels).astype(np.uint8)

    return labels



def getTestDataset(test, test_label, size=250):
    '''
    Arguments: whole test data, test label,
    return randomized test data, test label of 'size'
    '''
    from numpy import array
    from random import shuffle

    assert test.shape[0] == test_label.shape[0]

    idx = list(range(test.shape[0]))
    shuffle(idx)
    idx = idx[:size]
    accuracy_x, accuracy_y = [], []
    for i in idx:
        accuracy_x.append(test[i])
        accuracy_y.append(test_label[i])

    return array(accuracy_x), array(accuracy_y)


def plot_random_spec_img(pic, true_label):
    '''
    Take first hyperspectral image from dataset and plot spectral data distribution
    Arguements pic = list of images in size (?, height, width, bands), where ? represents any number > 0
                true_labels = lists of ground truth corrospond to pic
    '''
    pic = pic[0]  #Take first data only
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from numpy import mean, argmax

    print("Image Shape: " + str(pic.shape) )
    print("Label of this image is -> " + str(true_label[0] ) )

    title = argmax(true_label[0], axis=0)
    # Calculate mean of all elements in the 3d element
    mean_value = mean(pic)
    # Replace element with less than mean by zero
    pic[pic < mean_value] = 0
    
    x = []
    y = []
    z = []
    # Coordinate position extractions
    for z1 in range(pic.shape[0]): 
        for x1 in range(pic.shape[1]):
            for y1 in range(pic.shape[2]):
                if pic[z1,x1,y1] != 0:
                    z.append(z1)
                    x.append(x1)
                    y.append(y1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('True class = '+ str(title))
    ax.scatter(x, y, z, color='#0606aa', marker='o', s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Spectral Label')
    ax.set_zlabel('Y Label')
    plt.show()


def GroundTruthVisualise(data):
    from matplotlib.pyplot import imshow, show, colorbar, set_cmap
    imshow(data)
    set_cmap('tab20b')
    colorbar()
    show()


# Arguement: data = 3D image in size (h,w,bands)
def plotStatlieImage(data, bird=False):
    from matplotlib.pyplot import imshow, show, subplots, axis, figure
    print('\nPlotting a band image')
    fig, ax = subplots(nrows=3, ncols=3)
    i = 1
    for row in ax:
        for col in row:
            i += 11
            if bird:
                col.imshow(data[i,:,:])
            else:
                col.imshow(data[:,:,i])
            axis('off')
    show()


def showClassTable(number_of_list, title='Number of samples'):
    import pandas as pd 
    print("\n+------------Show Table---------------+")
    lenth = len(number_of_list)
    column1 = range(1, lenth+1)
    table = {'Class#': column1, title: number_of_list}
    table_df = pd.DataFrame(table).to_string(index=False)
    print(table_df)   
    print("+-----------Close Table---------------+")



# This section here is for debugs only
if __name__ == '__main__':
    pass
