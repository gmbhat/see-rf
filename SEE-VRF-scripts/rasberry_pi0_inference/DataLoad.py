import pickle
import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import copy

class LoadData:

   
    def __init__(self):
       
        # self.filename = filename
        self.data = None
        self.labels_array = None
        self.n_window = None
       
       
    def Read(self , datasetName = None):
   
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, default='EMGPhysical', required=False)
        parser.add_argument('--model_type', type=str, help="model", required=False, default="classic")
        args = parser.parse_args()
       
        if datasetName is None:
            dataset_name = args.dataset_name  
        else:
            dataset_name = datasetName
           
        model_type = args.model_type
        file_path = f'Datasets/{dataset_name}_dataLabels.pkl'
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)
        self.data = data_dict['data']
        self.labels_array = data_dict['labels']

        self.n_window, n_channel, n_data = self.data.shape
        unique_numbers_set = set(self.labels_array)
        num_activities = len(unique_numbers_set)
       
        # Aug_index = self.labels_array[(self.labels_array == 3)] #Augmenting R windows
        # aug_data = self.data[(self.labels_array == 3),:]
        # self.labels_array = np.append(self.labels_array,Aug_index)
        # self.data = np.append(self.data,aug_data,0)
       
        m,n = self.data.shape[::2]                                                      
        self.n_window, n_channel, n_data = self.data.shape
       
       
    def SplitData(self):
       
        X = self.GetData()
        y = self.GetLabel()

        lst = list(range(0, self.GetWindow()))
       
       
        # First split: 80% (Train + Validation) and 20% (Test)
        X_temp, X_test_ind, l_temp, l_test = train_test_split(lst, y, test_size=0.20, random_state=42)

        # Second split: Split 80% into 60% (Train) and 20% (Validation)
        X_train_ind, X_val_ind, l_train, l_val = train_test_split(X_temp, l_temp, test_size=0.25, random_state=42)  # 0.25 * 80% = 20%

        # data
        self.X_train = X[X_train_ind,:,:]
        self.X_test =  X[X_test_ind,:,:]
        self.X_val =   X[X_val_ind,:,:]

        # labels
        self.y_train = y[X_train_ind]
        self.y_test = y[X_test_ind]
        self.y_val = y[X_val_ind]

       
        n_window, n_channel, n_data = X.shape
       
        # flatten the data
       
        input_data_RF_train = copy.deepcopy(self.X_train)
        input_data_RF_train = np.swapaxes(input_data_RF_train, 1, 2)  
        self.train_X_flatten = input_data_RF_train.reshape(len(self.y_train), n_channel*n_data)

        input_data_RF_val = copy.deepcopy(self.X_val)
        input_data_RF_val = np.swapaxes(input_data_RF_val, 1, 2)
        self.val_X_flatten = input_data_RF_val.reshape(len(self.y_val), n_channel*n_data)

        input_data_RF_test = copy.deepcopy(self.X_test)
        input_data_RF_test = np.swapaxes(input_data_RF_test, 1, 2)
        self.test_X_flatten = input_data_RF_test.reshape(len(self.y_test), n_channel*n_data)
               
               
    def GetData(self):
        return self.data
   
    def GetLabel(self):
        return self.labels_array
   
    def GetWindow(self):
        return self.n_window
   
    def GetTrainX(self):
        return self.train_X_flatten
   
    def GetValX(self):
        return self.val_X_flatten
   
    def GetTestX(self):
        return self.test_X_flatten
   
    def GetYtrain(self):
        return self.y_train
   
    def GetYtest(self):
        return self.y_test
   
    def GetYval(self):
        return self.y_val