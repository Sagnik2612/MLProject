## This is our main ML Project

We will go through the entire process of creating and editing our ML project over here and keep track of the changes in README.md

### 1.setup.py file
If we want to have the entire the whole application as a package, we make the setup.py file. The necessary information of the package is addressed in the setup.py file- eg. the Owner of the package, his/her email id., copyright,maintainer, description,author, url,license, version,python recquirement and more.

### 2.Automating and making better python project templates with cookiecutter library
Install the cookiecutter package to automate the otherwise manually tedious project structure/template making in python.The code is available on cookiecutter's homepage.We will execute the project structure building manually as well to get a good idea how to execute this step both ways.For this we will write a python code of our own that will produce the project structure.

An end-end data science projects has the main aim to build pipelines.Pipelining has two major steps--
(All the arrowed steps below are called the components which will have a components folder)

1.Training Pipeline-
Data Reading from a Database/anywhere else-->Follow lifecycle of a Data Science Project(Train-Test-Split-->EDA-->feat. Eng.)-->Train the Model-->Get the Model ready-->Model Evaluation-->Model Monitoring

2.Prediction Pipeline-
Take Input from Client end-->Give it to the Model (model deployed somewhere else or in API form)-->Give a valid output
