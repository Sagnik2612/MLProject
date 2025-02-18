#Now we will call logger file from src MLProject sub folder in app.py

from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_ingestion import DataIngestionConfig
from src.MLProject.components.data_transformation import DataTransformationConfig
from src.MLProject.components.data_transformation import DataTransformation
import sys

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
        dat_trans.initiate_dat_trans(train_data_path,test_data_path) 


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    
    #Exception will capture error message "e" 
    # and all error details,error messages will be in sys
    


#Now go to the logs folder and check every .log file for the 
#given moment execution and message of the code