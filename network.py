from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from argparse import ArgumentParser
from model import *
import tensorflow as tf
from helper import showClassTable, maybeExtract
import os

GPU_DEVICE_IDX = '3'
model_directory = os.path.join(os.getcwd(), 'Trained_model/')

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines', help='Indian_pines or Salinas or KSC or Botswana')
parser.add_argument('--epochs', type=int, default=650)
parser.add_argument('--device', type=str, default='CPU')

number_of_band = {'Indian_pines': 4, 'Salinas': 8, 'KSC': 8, 'Botswana': 1}

def mbnet(statlieImg, prob, HEIGHT, WIDTH, CHANNELS, N_PARALLEL_BAND, NUM_CLASS):

    print(statlieImg)
    sequence = {}
    sequence['inputLayer'] = tf.reshape(statlieImg, [-1, HEIGHT, WIDTH, CHANNELS])
    # Block 1 Conv1 layer
    with tf.variable_scope('conv1'):
        layer = sequence['inputLayer']
        layer = create_conv_2dlayer(input=layer,
                                    num_input_channels=CHANNELS,
                                    filter_size=3,
                                    num_output_channel=CHANNELS,
                                    stride=1,
                                    relu=True, pooling=False)
        sequence['conv1'] = layer

    print(layer, 'layer1')
    with tf.variable_scope('parallelProcess-Block2'):
        layer = sequence['conv1']

        with tf.variable_scope('reshape2d'):
            layer = tf.reshape(layer, [-1, CHANNELS, 9, 1])

        with tf.variable_scope('transpose'):
            layer = tf.transpose(layer, [0, 2, 1, 3])

        with tf.variable_scope('split'):
            chunked_layer = tf.split(layer, num_or_size_splits=N_PARALLEL_BAND, axis=2)

        segment = chunked_layer[0]

        print(segment)
        with tf.variable_scope('layer1'):
            layer1 = create_conv_2dlayer(input=segment,
                                         num_input_channels=1,
                                         filter_size=3,
                                         num_output_channel=1,
                                         stride=1,
                                         relu=True, pooling=False)

        with tf.variable_scope('layer2'):
            layer2 = create_conv_2dlayer(input=layer1,
                                         num_input_channels=1,
                                         filter_size=3,
                                         num_output_channel=2,
                                         stride=1,
                                         relu=True, pooling=False)

        with tf.variable_scope('layer3'):
            layer3 = create_conv_2dlayer(input=layer2,
                                         num_input_channels=2,
                                         filter_size=3,
                                         num_output_channel=4,
                                         stride=1,
                                         relu=True, pooling=False)

        with tf.variable_scope('layer4'):
            layer4 = create_conv_2dlayer(input=layer3,
                                         num_input_channels=4,
                                         filter_size=3,
                                         num_output_channel=4,
                                         stride=1,
                                         relu=True, pooling=False)

            layer5, _ = flatten_layer(layer4)
            stack = tf.concat([layer5], axis=3)

        # Parameter sharing
        tf.get_variable_scope().reuse_variables()

        for l in chunked_layer[1:]:

            with tf.variable_scope('layer1'):
                layer1 = create_conv_2dlayer(input=l,
                                             num_input_channels=1,
                                             filter_size=3,
                                             num_output_channel=1,
                                             stride=1,
                                             relu=True, pooling=False)

            with tf.variable_scope('layer2'):
                layer2 = create_conv_2dlayer(input=layer1,
                                             num_input_channels=1,
                                             filter_size=3,
                                             num_output_channel=2,
                                             stride=1,
                                             relu=True, pooling=False)

            with tf.variable_scope('layer3'):
                layer3 = create_conv_2dlayer(input=layer2,
                                             num_input_channels=2,
                                             filter_size=3,
                                             num_output_channel=4,
                                             stride=1,
                                             relu=True, pooling=False)

            with tf.variable_scope('layer4'):
                layer4 = create_conv_2dlayer(input=layer3,
                                             num_input_channels=4,
                                             filter_size=3,
                                             num_output_channel=4,
                                             stride=1,
                                             relu=True, pooling=False)

            layer5, _ = flatten_layer(layer4)

            stack = tf.concat([stack, layer5], axis=1)

        sequence['parallel_end'] = stack

    with tf.variable_scope('dense1'):
        layer = sequence['parallel_end']
        layer, number_features = flatten_layer(layer)
        layer = fully_connected_layer(input=layer,
                                      num_inputs=number_features,
                                      num_outputs=120,
                                      activation='relu')
        layer = tf.nn.dropout(x=layer, keep_prob=prob)
        sequence['dense1'] = layer

    with tf.variable_scope('dense3'):
        layer = sequence['dense1']
        layer = fully_connected_layer(input=layer,
                                      num_inputs=120,
                                      num_outputs=NUM_CLASS)
        sequence['dense3'] = layer

    y_predict = tf.nn.softmax(sequence['dense3'])
    sequence['class_prediction'] = y_predict
    sequence['predict_class_number'] = tf.argmax(y_predict, axis=1)

    return sequence


def main(opt):

    if opt.device == 'GPU':
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE_IDX

    # Load MATLAB data that contains data and labels
    PATCH = 5
    TRAIN, VALIDATION, TEST = maybeExtract(opt.data, PATCH)

    # Extract data and label from MATLAB file
    training_data, training_label = TRAIN['train_patch'], TRAIN['train_labels']
    validation_data, validation_label = VALIDATION['val_patch'], VALIDATION['val_labels']
    test_data, test_label = TEST['test_patch'], TEST['test_labels']

    print('\nData shapes')
    print('training_data shape' + str(training_data.shape))
    print('training_label shape' + str(training_label.shape) + '\n')
    print('validation_data shape' + str(validation_data.shape))
    print('validation_label shape' + str(validation_label.shape) + '\n')
    print('test_data shape' + str(test_data.shape))
    print('test_label shape' + str(test_label.shape) + '\n')

    SIZE = training_data.shape[0]
    HEIGHT = training_data.shape[1]
    WIDTH = training_data.shape[2]
    CHANNELS = training_data.shape[3]
    N_PARALLEL_BAND = number_of_band[opt.data]
    NUM_CLASS = training_label.shape[1]
    EPOCHS = opt.epochs

    # Used for printing class number
    report_label = []
    for n in range(1, NUM_CLASS + 1):
        report_label.append('Class ' + str(n))

    graph = tf.Graph()
    with graph.as_default():
        # Define Model entry placeholder
        img_entry = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNELS], name='img_entry')
        img_label = tf.placeholder(tf.uint8, shape=[None, NUM_CLASS], name='img_label')

        # Get true class from one-hot encoded format
        image_true_class = tf.argmax(img_label, axis=1, name="img_true_label")

        # Dropout probability for the model
        prob = tf.placeholder(tf.float32)

        # Model definition
        model = mbnet(img_entry, prob, HEIGHT, WIDTH, CHANNELS, N_PARALLEL_BAND, NUM_CLASS)

        # Cost Function
        final_layer = model['dense3']

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                                       labels=img_label)
        cost = tf.reduce_mean(cross_entropy)

        # Optimisation function
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

        # Model Performance Measure
        with tf.name_scope('accuracy'):
            predict_class = model['predict_class_number']
            correction = tf.equal(predict_class, image_true_class)
        accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))

        # Checkpoint Saver
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            '''
            writer = tf.summary.FileWriter("network-logs/", session.graph)
            '''
            '''
            if os.path.isdir(model_directory):
                saver.restore(session, 'Trained_model/')
            '''
            session.run(tf.global_variables_initializer())

            def train(num_iterations, train_batch_size):
                maxValidRate = 0
                location = 0
                for i in range(num_iterations + 1):
                    print('Optimization Iteration: ' + str(i))
                    for x in range(int(SIZE / train_batch_size) + 1):

                        train_batch = training_data[x * train_batch_size: (x + 1) * train_batch_size]
                        train_batch_label = training_label[x * train_batch_size: (x + 1) * train_batch_size]
                        feed_dict_train = {img_entry: train_batch, img_label: train_batch_label, prob: 0.5}
                        session.run(optimizer, feed_dict=feed_dict_train)

                    # Run Validation Accuracy for every 15 epochs
                    if i % 15 == 0:
                        acc = session.run(accuracy, feed_dict={img_entry: validation_data, img_label: validation_label,
                                                               prob: 1.0})
                        print('Model Performance, Validation accuracy: ' + str(acc * 100))


                        # Run test data to check accuracy
                        test_x, test_y = test_data, test_label
                        feed_dict_validate = {img_entry: test_x, img_label: test_y, prob: 1.0}
                        class_pred = np.zeros(shape=test_x.shape[0], dtype=np.int)
                        class_pred[:test_x.shape[0]] = session.run(model['predict_class_number'],
                                                                  feed_dict=feed_dict_validate)
                        class_true = np.argmax(test_y, axis=1)
                        report = classification_report(class_true, class_pred, target_names=report_label, digits=5)
                        correct = (class_true == class_pred).sum()
                        accuracy_test = float(correct) / test_x.shape[0]
                        if accuracy_test > maxValidRate:
                            maxValidRate = accuracy_test
                            location = i

                        print('Maximum Test accuracy: \t' + str(maxValidRate * 100) + '% at epoch' + str(location))
                        print('Overall Accuracy at Test: \t' + str(accuracy_test * 100) + '%')
                        print('Confusion matrix')
                        con_mat = confusion_matrix(class_true, class_pred)
                        print(con_mat)
                        print(report)

            def test(test_iterations=1, test_data=test_data, test_label=test_label):
                print('-----Running Test set-------')
                assert test_data.shape[0] == test_label.shape[0]

                resultSize = test_data.shape[0]

                y_predict_class = model['predict_class_number']
                test_img_batch, test_img_label = test_data, test_label

                # OverallAccuracy, averageAccuracy and accuracyPerClass
                overAllAcc, avgAcc, averageAccClass = [], [], []
                for i in range(test_iterations):
                    feed_dict_test = {img_entry: test_img_batch, img_label: test_img_label, prob: 1.0}

                    class_pred = np.zeros(shape=resultSize, dtype=np.int)
                    class_pred[:resultSize] = session.run(y_predict_class, feed_dict=feed_dict_test)

                    class_true = np.argmax(test_img_label, axis=1)
                    conMatrix = confusion_matrix(class_true, class_pred)

                    # Calculate recall score across each class
                    classArray = []
                    for c in range(len(conMatrix)):
                        recallSoc = conMatrix[c][c] / sum(conMatrix[c])
                        classArray += [recallSoc]
                    averageAccClass.append(classArray)
                    avgAcc.append(sum(classArray) / len(classArray))
                    overAllAcc.append(accuracy_score(class_true, class_pred))

                averageAccClass = np.transpose(averageAccClass)

                meanPerClass = np.mean(averageAccClass, axis=1)

                showClassTable(meanPerClass, title='Class accuracy')
                print('Average Accuracy: ' + str(np.mean(avgAcc)))
                print('Overall Accuracy: ' + str(np.mean(overAllAcc)))

            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('Trainable parameters: ' + '\033[92m' + str(total_parameters) + '\033[0m')

            # Train model
            train(num_iterations=EPOCHS, train_batch_size=50)
            saver.save(session, model_directory)

            test(test_iterations=1)

            print('End session: ' + str(opt.data))


if __name__ == '__main__':
    option = parser.parse_args()
    main(option)
