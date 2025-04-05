#Data Transformation
#This section is mostly about Feature Engineering+EDA on the train+test data
#Aim-->
#Take the train-test-split data from data ingestion-->Do Feat.Engin. on it-->Get the output pickle file of 
#Feat Engin.


import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.compose import ColumnTransformer
#
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
# SimpleImputer To handle categorical features and missing values
from sklearn.pipeline import Pipeline
#Pipeline to make pipelines

from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
import os #to save the feat. engin.in pickle file

#Similar structure to Data Ingestion
#First create directory structure/paths in a config class 
#Path of th pickle file where the feat.engin. gets saved in 
#Then the main Feat.Engin. part

from src.MLProject.utils import save_obj

@dataclass

class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifact','preprocessor.pkl')
    # preprocessor pickle file path stored in preprocessor_obj_path

class DataTransformation:
    def __init__(self):
        self.dat_trans_config=DataTransformationConfig() #initializing

    def data_transf_obj(self):
        #It's functionality is to perform feature engin.
        try:
            num_col=["reading_score","writing_score"] #numerical columns
            cat_col=["gender",
                     "race_ethnicity",
                     "parental_level_of_education",
                     "lunch",
                     "test_preparation_course"
                     ]
            #StandardScaler will be applied to Numerical Features
            #OneHotEncoder will be applied to Categorical Variables

            #Pipeline-


            #In the future if the dataset inputs new data then we also need
            #to handle the missing values-Thats why we create a pipeline 
            #The Pipeline class in scikit-learn is a powerful tool designed to streamline the 
            # machine learning workflow. It allows you to chain together multiple steps, such as data 
            # transformations and model training, into a single, cohesive process. 
            # This not only simplifies the code but also ensures that the same sequence of steps is 
            # applied consistently to both training and testing data, thereby reducing the risk of data 
            # leakage and improving reproducibility.
            #The process of transforming raw data into a model-ready format often involves a series of steps, 
            # including data preprocessing, feature selection, and model training. 
            # Managing these steps efficiently and ensuring reproducibility can be challenging.
            #Pipeline allows you to sequentially apply a list of transformers to preprocess the data and,
            # if desired, conclude the sequence with a final predictor for predictive modeling.
            # Intermediate steps of the pipeline must be transformers, that is, they must implement 
            # fit and transform methods. 
            #The final estimator only needs to implement fit. 
            # The transformersin the pipeline can be cached using memory argument.
            #The purpose of the pipeline is to assemble several steps that can be cross-validated
            #  together while setting different parameters. For this, it enables setting parameters 
            # of the various steps using their names and the parameter name separated by a '__', as in the 
            # example below. A step’s estimator may be replaced entirely by setting the parameter with its name 
            # to another estimator, or a transformer removed by setting it to 'passthrough' or None.

            num_pipe=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                     ("scalar",StandardScaler())]
                                     )

            #imputer fills in the missing values in the newly added 
            # data with a strategy(maybe mean,median or mode)
            #If the data is normally distributed with very less or no outliers,
            #  median is used as the strategy

            #StandardScalar or z-score Standardizes the numerical data

            cat_pipe=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                     ("one_hot_encoder",OneHotEncoder()),
                                     ("standard_scalar",StandardScaler(with_mean=False))
                                     ])
            
            #z = (x - u) / s
            #where u is the mean of the training samples or zero if with_mean=False, and s 
            # is the standard deviation of the training samples or one if with_std=False.

            logging.info(f"Categorical Columns:{cat_col}")
            logging.info(f"Numerical Columns:{num_col}")

            #At the end of the day we need to combine the 2 pipelines-numerical and categorical
            #The ColumnTransformer class does the job which takes the individual pipeline details 
            #in a list

            preprocessor=ColumnTransformer([ ("numerical_pipeline",num_pipe,num_col),
                                            ("categorical_pipeline",cat_pipe,cat_col)])
            

            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_dat_trans(self,train_path,test_path):
        #Inputs the output(train and test path) of the Data Ingestion
        #We will take the train and test files and transform it over here

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            #Now on calling the data_transf_obj() function, all the Feature Engin.
            # tasks should be performed

            prep_obj=self.data_transf_obj()

            target_col="math_score"
            num_col=["reading_score","writing_score"]

            #Dividing train dataset into independent and dependent features

            indep_feat_train=train_df.drop(columns=[target_col],axis=1)
            target_feat_train=train_df[target_col]

            #Dividing test dataset into independent and dependent features

            indep_feat_test=test_df.drop(columns=[target_col],axis=1)
            target_feat_test=test_df[target_col]

            logging.info("Applying Pre-procesing on train and test dataframes")

            indep_feat_train_arr=prep_obj.fit_transform(indep_feat_train)
            indep_feat_test_arr=prep_obj.transform(indep_feat_test)
            #transform() and NOT fit_transform() is used for test array indep. features to 
            #avoid data leakage

            #Now we will array join columnwise the transformned
            #independant features and the target features
            
            #Join((Indep. train array),array form of(target train dataframe)) columnwise
            #Join((Indep. test array),array form of(target test dataframe)) columnwise

            #This is done using the numpy np.c_[1st array,2nd array] feature
            #c_ means concatenate/join

            train_arr=np.c_[indep_feat_train_arr,np.array(target_feat_train)]
            test_arr=np.c_[indep_feat_test_arr,np.array(target_feat_test)]

            logging.info(f"Saved preprocessing object")

            #The Preprocessing object has to be saved in a pickle file
            #As it is a common functionality, we write its code in utils.py
            #That's why we will call the save_obj() function from utils that will
            #dump the file path(here artifacts) and the functionality (object itself) into save_obj(
            #save_obj(file_path to be saved in,functionality(preprocessor object)) and get back
            #the pickle file
            #filepath to be saved in comes from preprocessor_obj_path of the 
            #DataTransformationConfig class

            save_obj(file_path=self.dat_trans_config.preprocessor_obj_path,obj=prep_obj)

            #return train array,test array,preprocessor object file path
            #so that the next stage model trainer can use them as inputs

            return(train_arr,
                   test_arr,
                   self.dat_trans_config.preprocessor_obj_path)

#We can run this on training pipeline file or app.py 















        except Exception as e:
            raise CustomException(e,sys)
