from setuptools import find_packages,setup
from typing import List

# find_packages maps the entire folder structure of the application
#and wherever it finds folders and other packages, it converts
#them into a package each

#we define the get_recquirements() function over here
#We have encountered "-e ." in req.txt file
#What it does is- as req.txt gets read in setup.py, as soon as -e . gets encountered
#without even calling setup.py, we can call setup.py by directly calling
#req.txt.
#If we write pip install -r req.txt, when the python package gets installed,
#when it sees -e . it automatically triggers setup.py file
#But on running this, even "-e ." gets read but this is NOT a package
#Hence we need to remove it whenever it is encountered

E_DOT="-e ."

def get_requirements(file_path:str)-> List[str]:
    #file path taken as input in string format
    #and get_req() will return a list of strings (list of reqs from reqs.txt)

    req=[] #to store the requirements
    #Next the open() function will open the given file path 
    with open(file_path) as file_obj:
        req=file_obj.readlines() #reads the file path line by line
        #Between each line in the req.txt file is a "\n" string (invisible)
        #This new line thing "\n" should be removed
        #we do a list comprehension to remove this "\n" 
        # and replace it with a blank " "
        req=[r.replace("\n","") for r in req]

        #removing "-e ." from req.txt
        if E_DOT in req:
            req.remove(E_DOT)

    return req





#The setup function will contain the entire project information

#setup function



setup(
name="MLProject",
version="0.0.1",
author="Sagnik Bhattacharjee",
author_email="sagnikbhatt26@gmail.com",
packages=find_packages(),
install_requires=get_requirements("recquirements.txt")
#instead of maually putting in the recquired package names
#we will directly access it from requirements.txt
# get_requirements() reads the req.txt file and fetches all the package
# names as a python list 
)

#On running the setup.py file, the src (source folder) folder
#where the entire project will be executed will become a package of its own