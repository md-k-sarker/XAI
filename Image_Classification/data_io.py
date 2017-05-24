'''
Created on May 10, 2017

@author: sarker
'''

import csv
import numpy as np


class ManipulateData(object):
    '''
    Manipulate data
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def get_datamatrix_from_csv(self, fileName):
        dataMatrix = np.array(
            list(csv.reader(open(fileName, "r+", encoding="utf-8"), delimiter=',')))

        return dataMatrix

    def shuffle_data(self, dataMatrix):
        dataMatrix = np.random.permutation(dataMatrix)
        return dataMatrix

    def convert_data_to_float(self, dataMatrix):
        dataMatrixAsfloat = [[np.float128(eachVal)
                              for eachVal in row] for row in dataMatrix]
        return dataMatrixAsfloat

    def convert_data_to_zero_one(self, dataList):
        dataAsZeroOne = []
        for eachVal in dataList:
            if(eachVal == 'N'):
                dataAsZeroOne.append(0)
            else:
                dataAsZeroOne.append(1)

        return dataAsZeroOne

    def get_normalized_data(self, dataMatrix):

        dataMatrix = np.array(dataMatrix)
        normalizedDatas = []

        for i in range(0, dataMatrix.shape[1]):

            columnData = dataMatrix[:, i]
            _mean = np.mean(columnData)
            _max = np.max(columnData)
            _min = np.min(columnData)

            normalizedData = []

            for eachData in columnData:
                normalizedData.append((eachData - _mean) / (_max - _mean))

            normalizedDatas.append(normalizedData)

        return np.array(normalizedDatas)

    def get_vectorized_class_values(self, classes):
        OutputLayerNoOfNeuron = len(set(classes))

        outputVector = np.zeros((len(classes), OutputLayerNoOfNeuron))

        for i in range(0, len(classes)):
            if(classes[i] == 1):
                outputVector[i] = [1, 0, 0]
            elif(classes[i] == 2):
                outputVector[i] = [0, 1, 0]
            elif(classes[i] == 3):
                outputVector[i] = [0, 0, 1]

        return outputVector

    def get_vectorized_class_values_from_yes_no(self, classes):
        OutputLayerNoOfNeuron = len(set(classes))

        outputVector = np.zeros((len(classes), OutputLayerNoOfNeuron))

        for i in range(0, len(classes)):
            if(classes[i] == 'N'):
                outputVector[i] = [0, 1]
            elif(classes[i] == 'R'):
                outputVector[i] = [1, 0]

        return outputVector

    def zero_one_class_counter(self, dataList):
        classZero = 0
        classOne = 0
        for i in dataList:
            if(i == 0):
                classZero += 1
            elif(i == 1):
                classOne += 1
        return classZero, classOne

    def spliting_to_n_fold(self, l, n):
        n = max(1, n)
        return (l[i:i + n] for i in range(0, len(l), n))

    def split_train_and_test_data(self, data, ratio):
        trainData = data[:int(len(data) * ratio)]
        testData = data[int(len(data) * ratio):]
        return trainData, testData

    def split_train_validate_and_test_data(self, inputData, outputVector, ratio1, ratio2, ratio3):
        dataLength = len(inputData)
        if(ratio1 + ratio2 + ratio3 != 1):
            print('Sum of ratio must be 1')
            return 'Sum of ratio must be 1'
        trainInputData = inputData[:int(dataLength * ratio1)]
        trainOutputVector = outputVector[:int(dataLength * ratio1)]

        validationInputData = inputData[int(
            dataLength * ratio1): int(dataLength * (ratio1 + ratio2))]
        validationOutputVector = outputVector[int(
            dataLength * ratio1): int(dataLength * (ratio1 + ratio2))]

        testInputData = inputData[int(dataLength * (ratio1 + ratio2)):]
        testOutputVector = outputVector[int(dataLength * (ratio1 + ratio2)):]

        return trainInputData, trainOutputVector, validationInputData, validationOutputVector, testInputData, testOutputVector


if __name__ == '__main__':
    pass
