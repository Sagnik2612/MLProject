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
import pymysql
import pickle
import numpy as np

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






