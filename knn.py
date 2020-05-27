""" Knn.py
   This is a class to make an instance of a Knn classifier.
    Author: Alan Rocha González
    Email: alan.rocha@udem.edu
    Institution: Universidad de Monterrey
    Created: Thursday 14th, 2020
"""
# Importing necessary libraries
import numpy as np
import operator
import tensorflow as data_set_mnist
import sys


# Knn class
class Knn(object):
    def __init__(self, k=3):
        """
        INPUT: data set path & K
        OUTPUT: Knn instance
        """
        """
        Construction Class method. This method set all attributes of the class
        and call the load_data method.
        
        """
        self.x_data = None
        self.y_data = None
        self.x_testing_data = None
        self.y_testing_data = None
        self.x_testing_data_unscaled = None
        self.k = k
        self.mean = []
        self.std = []
        self.load_data()
    


    def load_data(self):
        """ 
        load data from comma-separated-value (CSV) file and set
        x_data, y_data, y_testing_data, x_testing_data.
        """
        """
        INPUT: path_and_filename: the csv file name
        OUTPUT: None
        """
        # Opens file

        try:
            (x_train, y_train), (x_test, y_test) = data_set_mnist.keras.datasets.mnist.load_data()
            x_train = x_train[:50000,:,:]
            x_test = []
            y_train = y_train[:50000]
            y_test  = []
        except IOError:
            print ("Error: El archivo no existe")
            exit(0)


        x_training = []
        for x in x_train:
            x_training.append(x.flatten())
        self.x_data = np.array(x_training, dtype=float)

        x_testing = []
        for x in x_test:
            x_testing.append(x.flatten())
        self.x_testing_data = np.array(x_testing, dtype=float)

        self.y_data = y_train
        self.y_testing_data = y_test


    def predict(self, x0):
        """ 
        Method to predict x0 on the training set
        """
        """
        INPUT: x0 Numpy array with the N characteristics
        OUTPUT: Prediction, Prob. Diabetes, Prob. No Diabetes
        """

        # Get N samples
        N = self.x_data.shape[0]

        # Set dictonary
        distances = {}

        # Calculate all euclidean distances
        for x in range(0, N):
            distances[x] = self.__compute_euclidean_distance(self.x_data[x]-x0)

        # Sorting distances
        distances = sorted(distances.items(), key=operator.itemgetter(1))

        return self.__compute_conditional_probabilities(distances)


    def get_confusion_matrix(self):
        """ 
        Method to get Testing point features and confusion matrix of the 
        """
        """
        INPUT: None
        OUTPUT: None
        """
        # Initiate variables to the confusion matrix
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        cont = 0
        i = 0.0
        # Evaluate x_testing
        for x,y in zip(self.x_testing_data, self.y_testing_data):
            i += 1.0
            print((i/len(self.x_testing_data))*100)
            prediction = self.predict(x)
            if(prediction == y):
                cont += 1
            print(str((cont/i)*100))

        print(cont)
        print(cont/len(self.x_testing_data)*100)
        


    def set_k(self, k):
        """ 
        Method to set K attribute
        """
        """
        INPUT: k a integer method
        OUTPUT: None
        """
        self.k = k
        # print("Testing K = "+str(k))



    # Private methods
    # ----------------------------------------------------------------------------
    # Set up data methods
    def __data_frame(self, file): 
        """ shuffle data of the csv file """
        """
        INPUT: filename: the csv file name
        OUTPUT: Return the shuffled dataframe
        """
        df = pd.read_csv(file, header=0)
        # return the pandas dataframe
        return df.reindex(np.random.permutation(df.index))

    def __feature_scaling_operation(self, data, mean_value, std_value):
        """ standarize the x data and saves mean value & std value"""
        """
        INPUT: data: data from de data set that will be standarized (numpy array)
            mean_value: mean_value (float)
            std_value: standard variation value (float)
        OUTPUT: Returns de data set standarized, the mean value and std value
        """
        if mean_value == 0 and std_value == 0:
            std_value = data.std()
            mean_value = data.mean()
        scaling_scores = (data - mean_value) / std_value
        return scaling_scores, mean_value, std_value

    def __feature_scaling(self, x, data_type):
        """ Apply feature scaling for the data set """
        """
        INPUT: x: numpy array dataset
               data_type: string 
        OUTPUT: numpy array 
        """
        scaled_array = []

        if data_type == "training":

            for feature in x.T:
                dataScaled, meanX, stdX = self.__feature_scaling_operation(feature, 0, 0)
                scaled_array.append(np.array(dataScaled))
                self.mean.append(meanX)
                self.std.append(stdX)
        else:
            for feature,mean,std in zip (x.T, self.mean, self.std):
                dataScaled = self.__feature_scaling_operation(feature, mean, std)
                scaled_array.append(np.array(dataScaled[0]))

        return np.array(scaled_array).T



    # ----------------------------------------------------------------------------
    # Prediction Methods
    
    def __compute_euclidean_distance(self, eval_x):
        """ Apply euclidean distance for an array """
        """
        INPUT: eval_x: numpy array dataset
        OUTPUT: float euclidean distance
        """
        return np.sqrt(np.sum((eval_x)**2))

    def __compute_conditional_probabilities(self, distances):
        """ Define classification probabilities """
        """
        INPUT: distances: dictonary [key: position, value: distance]
        OUTPUT: Prediction, Prob. Diabetes, Prob. No Diabetes
        """
        results  = [0]*10
        for predict in range(0, self.k):
            element = distances[predict][0]
            data = self.y_data[element]
            results[data] += 1


        index, value = max(enumerate(results), key=operator.itemgetter(1))
        print(results)
        return int(index)



    # ----------------------------------------------------------------------------
    # Prints methods
    def __print_data_set(self, x_data, y_data, leyend):
        """ prints x and y data """
        """
        INPUT: x_data & y_data: numpy arrays, leyent: title (string)
        OUTPUT: None
        """
        print("\n")
        print("--"*23)
        print(leyend)
        print("--"*23)
        for x,y in zip(x_data, y_data):
            print(x, y)
        print("\n\n\n")
    
    def __print_unscaled_result(self, x_testing_data_unscaled, one, zero):
        """ prints x_testing_data_unscaled, Prob. Diabetes and Prob. No Diabetes  """
        """
        INPUT: x_testing_data_unscaled: numpy array, zero: float prob, one float prob
        OUTPUT: None
        """
        for characteristic in x_testing_data_unscaled:
            print(round(characteristic, 3), end="\t\t")
        print(str(one)+"\t"+str(zero))

    def __print_perfomance_metrics(self, tp, tn, fp, fn):
        """ Display confusion matrix and performance metrics"""
        """
        INPUT:  tp: True positive (count)
                tn = True negative (count)
                fp = False positive (count)
                fn = False negative (count)
        OUTPUT: NONE
        """
        #Prints confusion matrix
        print("\n")
        print("--"*23)
        print("Confusion Matrix")
        print("--"*23)
        print("\t\t\t\t\t\tActual Class")
        print("\t\t\t\t\tGranted(1)\tRefused(0)")
        print("Predicted Class\t\tGranted(1)\tTP: "+str(tp)+"\t\tFP: "+str(fp)+"")
        print("\t\t\tRefused(0)\tFN: "+str(fn)+"\t\tTN: "+str(tn)+"")
        print("\n")

        # Calculate accuracy
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        # Calculate precision
        precision = (tp)/(tp+fp)
        # Calculate recall
        recall = (tp/(tp+fn))
        # Calculate specifity
        specifity = (tn/(tn+fp))
        # Calculate f1 score
        f1 = (2.0*((precision*recall)/(precision+recall)))

        # Print performance metrics
        print("Accuracy:"+str(accuracy))
        print("Precision:"+str(precision))
        print("Recall:"+str(recall))
        print("Specifity:"+str(specifity))
        print("F1 Score: " + str(f1))
        print("\n\n")
