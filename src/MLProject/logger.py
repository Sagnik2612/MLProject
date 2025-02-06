import logging
import os
from datetime import datetime

# Always checkout the python logger documentation page 
# We need to setup default logging return messages,
# that needs a code structure of its own for the message format
# This will also include the datetime stamp of every single log.

#The log file format will be-

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#The Strftime() function is used to convert date and 
# time objects to their string representation. 
# It takes one or more inputs of formatted code and returns 
# the string representation in Python.
#from datetime import datetime
#now = datetime.now()
#formatted = now.strftime("%Y-%m-%d %H:%M:%S")
#print(formatted)
#output format- 2023-07-18 06:24:12

#We will also build the log folder path
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #path

#os.getcwd() gives current working directory(cwd) of whichever file
#I am running it from.
# The folder name is "logs" which will contain all the datetime info 
# and other info of each log
#Wherever the file/code is executed from, that files directory
#will then be considered as the working directory.It will extract
#that path,open the folder "logs" inside it and all the
#LOG_FILE s will be created inside it
#Now as we have the path, we will make a folder/directory out of it 
#using os.makedirs()

os.makedirs(log_path,exist_ok=True)

#Full directory/file path of the log file-->

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

#Now, there is a format whenever we are 
# setting logging.There is a config function that keeps 
# track of the log file path, format
#in the form of messages

logging.basicConfig(
                    filename=LOG_FILE_PATH, 
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    #Log message format
                    #line no.s-lineno.s of the files that are creating a problem
                    #names-names of the files that are craeting problems
                    #message- give the type/kind of error encountered or gives the
                    #positive message like-the file was created
                    level=logging.INFO
                    #logging.INFO information will be shown in 
                    #the %(levelname)s
                    )


