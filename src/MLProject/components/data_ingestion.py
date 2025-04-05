#Data Ingestion
#Data from the student Table in the College Database needs to be read
#MySQL Database-->Read it in local in my application-->Train Test Split-->We have the dataset

import os #For current path and folders
import sys #To handle Custom Exception here+Logging
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
from src.MLProject.utils import read_sql_data
import pandas as pd
from sklearn.model_selection import train_test_split
from src.MLProject.components.data_transformation import DataTransformationConfig
from src.MLProject.components.data_transformation import DataTransformation
from src.MLProject.components.data_model_trainer import ModTrainConfig,ModelTrainer


from dataclasses import dataclass 
#Whatever input parameters are recquired 
# can be quickly initialized with dataclass
#This module provides a decorator and functions for
#  automatically adding generated 
# special methods such as __init__() and __repr__() to
#  user-defined classes.
#@dataclasses.dataclass(*, init=True, repr=True, eq=True,
#  order=False, unsafe_hash=False, frozen=False,
#  match_args=True, kw_only=False, slots=False, weakref_slot=False)
#This function is a decorator that is used to add generated special methods to classes, as described below.

#The @dataclass decorator examines the class to find fields. 
# A field is defined as a class variable that has a type annotation. 
# With two exceptions described below, nothing in @dataclass examines 
# the type specified in the variable annotation.

#The order of the fields in all of the generated methods 
# is the order in which they appear in the class definition.
#If @dataclass is used just as a simple decorator with no parameters, 
# it acts as if it has the default values documented in this signature.

#eg-
#@dataclass
#class C:
#    ...

#@dataclass()
#class C:
#    ...

#@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False,
#           match_args=True, kw_only=False, slots=False, weakref_slot=False)
#class C:
#    ...

#@dataclass
#class C:
#    a: int       # 'a' has no default value
#    b: int = 0   # assign a default value for 'b'

#In this example, both a and b will be included in the 
# added __init__() method, which will be defined as:

#def __init__(self, a: int, b: int = 0):

#The parameters to @dataclass are:
#init- If true (the default), a __init__() method will be generated.
#If the class already defines __init__(), this parameter is ignored.
#repr,hash,eq


#Decorator-A function returning another function, usually 
# applied as a function transformation using the @wrapper syntax. 
# Common examples for decorators are classmethod() and staticmethod().
#The decorator syntax is merely syntactic sugar, 
# the following two function definitions are semantically equivalent:

#def f(arg):
#    ...
#f = staticmethod(f)

#@staticmethod
#def f(arg):
#    ...


#PATH OF ACTION-

#                    MySQL Data Source
#                  input-Dataframe/Table
#                            +
#(Path to save the Train Data/File+Path to store the Test data/File)
#                            |
#                            |
#                     Data Ingestion
#                            |
#                            |
#                     Output-2 files
#                           / \
#                          /   \
#                      Train    Test
#                      File     File


#We want to save all the train,test,raw data files in an artifact folder

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact','train.csv') 
    #string type path 
    #Artifact folder will store train,test and raw csv files 
    test_data_path:str=os.path.join('artifact','test.csv') 
    raw_data_path:str=os.path.join('artifact','raw.csv') 
    #raw file will have the entire data file with all records as well
    #We use @dataclass to predefine the params train_data_path,test and raw

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        #Initial data ingestion configs (paths of train,test,raw files)
        #called from DataIngestionConfig class

    def initiate_data_ingestion(self):
        #1st step-read data from MySQL Database
        try:
            df=pd.read_csv(os.path.join('notebook/data','stud.csv'),delimiter=",")
            #df=read_sql_data() #raw data
            logging.info("Reading completed from SQL Database")
            #make the artifact folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            #saves dataframe in csv format to the raw data path folder 
            # with no index
            #with a header
            train,test=train_test_split(df,test_size=0.2,random_state=42)
            #Save train and test datasets in their respective paths
            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as ex:
            raise CustomException(ex,sys)



#An .env or environment variable needs to be created 
#because the connection string,local host,password,user,root (database info) from 
#MySQL database cannot be directly accessed.
#This leads to major privacy issues.
#.env variable /file takes care of all of these.
#This is set in a different way in production.

#Reading of the data from the database will happen in the ,utils file

#Finally we will call the read_sql_data code from the utils file to this 
#data_ingestion file 


#Execution point of the file with if statement

if __name__=="__main__":
    #To test if the execution has started
    logging.info("The execution has started")
    #Now if we call this, whatever file structure code is in logger.py
    #will be called and the folder structure skeleton in 
    # logger.py  will be created 
    #Anytime in the future we want to see the logs, we will put
    #only this much line of code to see the loggings
    try:
        #data_ingestion_config=DataIngestionConfig() #initialized
        data_ingestion=DataIngestion()#initialized
        #DataIngestion itself calls DataIngestionConfig so no extra need
        #for calling it separately
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #dat_trans_config=DataTransformationConfig() #initialized
        dat_trans=DataTransformation() #initialized
        train_arr,test_arr,_=dat_trans.initiate_dat_trans(train_data_path,test_data_path) 

        # model trainer
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_mod_train(train_arr,test_arr)) #returns the r2 score of best fit model



    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    
    #Exception will capture error message "e" 
    # and all error details,error messages will be in sys
    


#Now go to the logs folder and check every .log file for the 
#given moment execution and message of the code