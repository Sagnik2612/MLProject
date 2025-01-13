#The template.py file will contain code that will build the entire project 
# structure manually

import os #We want to build generic folder and file paths so os is needed
from pathlib import Path #It explicitly makes and gives us paths
import logging 
#Logging is a means of tracking events that
# happen when some software runs. The software’s developer adds 
# logging calls to their code to indicate that certain events have occurred. 
# An event is described by a descriptive message which can optionally contain variable data
# (i.e. data that is potentially different for each occurrence of the event).
# Events also have an importance which the developer ascribes to the event; 
# the importance can also be called the level or severity.

logging.basicConfig(level=logging.INFO) #We want the logging info

project_name="MLProject"

#list_of_files is a list that will have the project folder structure
list_of_files=[
    #.github folder is needed because on deployment we will need 
    #github actions. ".gitkeep" is an indication that we are making 
    #a github folder that will be used later in deployment
    ".github/workflows/.gitkeep",
    #f-string allows you to format selected parts of a string
    #We want the keep MLProject inside src folder 
    #and to make it a package we write /__init__.py
    #init.py file stays inside the src 
    f"src/{project_name}/__init__py",
    #MLProjects folder will have components folder and pipelines folder
    #Components folder structure
    f"src/{project_name}/components/__init__py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/data_model_trainer.py",
    #The data validation code is available in config ML code structure
    #If needed can be accessed later on
    f"src/{project_name}/components/data_model_monitoring.py",
    #Pipelines folder
    f"src/{project_name}/pipelines/__init__py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    #Inside MLProject folder we need exception handling,
    # logging files as well
    #and .utils file
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    #Other files
    "app.py",
    "Dockerfile",
    "recquirements.txt",
    "setup.py",
    "main.py"

]

#The next step is to actually create the directories with file 
#operations and track the loggins

#os.path.split() method in Python is used
# to Split the path name into a pair, and tail. 
# Here, the tail is the last path name component
# and the head is everything leading up to that.
#eg-path='/home/user/Desktop/file.txt' 
#split into- head='/home/user/Desktop/' and
#tail='file.txt'
#The output of os.path.split(filepath) is a 
# filedirectory(filedir/head) and filename (tail)
#filedir,filename=os.path.split(filepath)

#The immediate goal is- if the filedir is NOT empty " "
#then make the directory for that specific file

#if the filepath does not exist or the path size 
# with (getsize(path) function)
#is 0 then we will open the file in write 'w' mode
#and simply pass because we are not appending anything
#When the pass statement is executed, nothing happens,
#but you avoid getting an error when empty code is not allowed
#The filepath in this case is already empty so 
# it will create an empty file

#Else the logging will show that the filename already exists

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if(filedir!=""):
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file {filename}")

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as fp:
            pass
            logging.info(f"Creating empty file {filepath}")

    else:
        logging.info(f"Filename {filename} already exists")

