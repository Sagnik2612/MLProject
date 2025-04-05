#Now we will call logger file from src MLProject sub folder in app.py
#we will be making a flask app over here 

from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
from flask import Flask,request,render_template
#To access any POST request 
#Flask’s request module is used to access data sent by the user (like form inputs, query parameters, JSON data, etc.).
#Getting data from a form submission (POST request).
#Retrieving query parameters from a URL (GET request).
#Accessing JSON data from an API request.

#render_template-Instead of returning raw text, you can use render_template to serve HTML pages.
#Separates backend logic from the frontend (HTML, CSS, JS).
#Allows passing dynamic data to an HTML template.
#Makes your app more scalable and maintainable.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle 
import sys
from src.MLProject.pipelines.prediction_pipeline import CustomData,PredictPipeline



application=Flask(__name__) #entry point of execution

app=application

#Creating a Route for a home page
@app.route('/')
def index():
    return render_template('index.html')


#@app.route('/') → When a user visits the homepage (/), this function runs.
#render_template('index.html') → Loads the index.html file (located in the templates/ folder).
# What happens?
#If a user opens the website, Flask renders index.html.

@app.route('/predictdata',methods=['GET','POST'])

#This route (/predictdata) handles both GET and POST requests.
#GET → Displays the input form.
#POST → Processes form data and makes predictions.


def predict_datapoint():

    if request.method=='GET':
       return render_template('home.html')
    #If the user visits /predictdata directly, Flask renders home.html (the form page).

    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score') or 0),
            writing_score=float(request.form.get('reading_score') or 0))
        
         #request.form.get('name') → Extracts form values.
         #Converts reading_score and writing_score to float (since they are numbers).
         #Stores the values in a CustomData object
         #Why use CustomData?
         #This class formats the new input data into a DataFrame so that the ML model can process it.
        pred_df=data.convert_data_as_dataframe() #fetching input data as a dataframe
        #Converts user input into a Pandas DataFrame.
        print(pred_df)
        print("Before Prediction")

        pred_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=pred_pipeline.predict(pred_df)
        #predict(pred_df) transforms the data and makes a prediction.
        print("after Prediction")

        return render_template('home.html', results=results[0]) #The predicted value (results[0]) 
        #is sent back to home.html.
        #The HTML page displays the prediction.

#What happens?-After submitting the form, the user sees their predicted result on the same page (home.html).

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        

#Runs the Flask app when executed directly.
#host="0.0.0.0" makes the app accessible on all network devices (useful for deployment).


#How does this work in action?--
#User Visits the Website (/)- Flask renders index.html.
#User Goes to /predictdata-  GET Request: Renders home.html (the input form).
#User Submits the Form
#POST Request:
#✅ Flask extracts form data.
#✅ Converts it into a Pandas DataFrame.
#✅ Uses the trained model to make a prediction.
#✅ Displays the result on home.html.









