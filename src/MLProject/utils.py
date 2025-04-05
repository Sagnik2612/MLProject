#Reading of the data from the database will happen in the ,utils file

#utils.py has generic functionality like reading from a database
#Python Utils is a collection of small Python functions and 
# classes which make common patterns shorter and easier. 
# It is by no means a complete collection but it has
#  served me quite a bit in the past and I will keep extending it.


import os #For current path and folders
import sys #To handle Custom Exception here+Logging
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
import pandas as pd
from dataclasses import dataclass 
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql
import pickle
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from copy import deepcopy
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

load_dotenv() #This will load all my environment variables
#(sql db info here)

host=os.getenv("host") #fetches every individual info from .env file
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")


#1st way-pymysql library-
# This package contains a pure-Python MySQL 
# client library
#connects to the MySQL database remotely or locally using Python
#The proper way to get an instance of this class is to call connect()
#  method. This method establishes a connection to the MySQL 
# database and accepts several arguments:
#host – Host where the database server is located
#user – Username to log in as
#password – Password to use.
#database – Database to use, None to not use a particular one.
#port – MySQL port to use, default is usually OK. (default: 3306)
#pymysql has inbuilt functions that will convert sql tables to 
# python dataframes 



#2nd way- mysql-connector-python -
#MySQL Connector/Python enables Python programs to access MySQL databases,
#  using an
#  API that is compliant with the Python Database API Specification v2.0.
#Converting parameter values back and forth between Python and MySQL
#  data types, for example Python datetime and MySQL DATETIME. 
# You can turn automatic conversion on for convenience, or off for 
# optimal performance.
#All MySQL extensions to standard SQL syntax.
#Protocol compression, which enables compressing the data stream 
# between the client and server.
#Connections using TCP/IP sockets and on Unix using Unix sockets.
#Secure TCP/IP connections using SSL.
#Self-contained driver. Connector/Python does not require the MySQL
#  client library or any Python modules outside the standard library.

#python-dotenv library
# We will read the database info like local host,password,db name etc 
# from .env file
#To read .end file we need another library "python-dotenv"


def read_sql_data():
    #First pick up sql db info from .env file
    #This function will return the table as a dataframe
    logging.info("Reading from SQL Database started")
    try:
        #Create the Database mydb
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection to Database established",mydb)
        #Read inside dataframe df with pd.read_sql_query("Select * from students",mydb)
        #format-pd.read_sql_query(The normal sql query you write in sql,database connection name)
        df=pd.read_sql_query("Select * from student",mydb)
        print(df.head())

        return df
    
    


    except Exception as e:
        raise CustomException(e,sys)
    


#The Preprocessing objet has to be saved in a pickle file
#As it is a common functionality, we write its code in utils.py


def save_obj(file_path,obj):
    try:
        #First make the directory to save the preprocessor pickle file
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
            # Dumped the functionality as an object in the pickle file



    except Exception as e:
        raise CustomException(e,sys)
    

def train_and_evaluate_models(models, params, X_train, y_train, X_test, y_test):

    #Takes in all the pre-processed, train-test-split data and all the ML models and hyperparams
    #Returns the best model,model score
    
    report = {}  # Dictionary to store R² scores
    
       
    for model_name, model in models.items():

        try:
            logging.info(f"🚀 Training model: {model_name}")

            # Ensure the model is an instance, not a class reference
            if isinstance(model, type):
                model = model()  # Instantiate if it's a class

            # Get hyperparameters for the model
            param_grid = params.get(model_name, {})

             # Apply GridSearchCV if parameters exist
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logging.info(f"✅ Best parameters for {model_name}: {gs.best_params_}")
            else:
                best_model = model  # Use original model if no hyperparameters
            # ✅ FIXED: Use `gs.best_estimator_` instead of creating a new instance
            
            best_model.fit(X_train,y_train)

            logging.info(f"✅ Best parameters for {model_name}: {gs.best_params_}")

            # ✅ Debugging: Check if the model is fitted
            try:
                check_is_fitted(best_model)
                logging.info(f"✅ {model_name} is successfully fitted!")
            except NotFittedError:
                logging.error(f"❌ {model_name} is NOT fitted after GridSearchCV!")

            # Make predictions
            y_test_pred = best_model.predict(X_test)


            # Calculate R² score
            r2 = r2_score(y_test, y_test_pred)
               
            logging.info(f"R² Score for {model_name}: {r2}")

            # Store in the report dictionary
            report[model_name] = r2
            print(f"✅ {model_name} R² Score: {report[model_name]}")

        
        except Exception as e:

             logging.error(f"Exception occurred in {model_name}: {e}")
             raise CustomException(f"Failed in {model_name}", sys) from e  # Wrap original error

    # Return the dictionary with model names and their corresponding R² scores
    return report 
    """
    except Exception as e:
        raise CustomException(e,sys)"
    """


#This load_obj() function is used to load a saved object (like a trained ML model or preprocessor) 
# from a file using pickle.

def load_obj(file_path):
    #Takes file_path as an input, which is the location of the saved object (e.g., "artifacts/model.pkl").
    #Example call-model = load_object("artifacts/model.pkl")
    #Here, "artifacts/model.pkl" is a saved ML model file.

    try:
        with open(file_path,"rb") as file_obj:
            #Opening the File in Read-Binary (rb) Mode
            #pickle.load(file_obj) reads the binary file and converts it back into a Python object.
            #The function returns the loaded object (e.g., a trained ML model).
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)






