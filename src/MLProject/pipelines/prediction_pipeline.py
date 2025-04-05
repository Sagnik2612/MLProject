#A web application will be made which will be interacting with model.pkl and preprocessor.pkl files
#wrt any input data given. The app will have a form that will take in the input data for student performance
#click the submit button, the given data will interact with preprocessor.pkl, model.pkl and then predict

import sys
import os
import pandas as pd 
from src.MLProject.exception import CustomException
from src.MLProject.utils import load_obj


class PredictPipeline:
    def __init__(self):
        pass

    #Defines a PredictPipeline class.
    #__init__() is an empty constructor (nothing happens when the object is created).


    def predict(self,features):
        #Defines predict(features) method → Takes input features (student details).
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join("artifact","preprocessor.pkl")
            #Finds model & preprocessor file paths (model.pkl, preprocessor.pkl).


            # Loading the Model and Preprocessor
            print("Before Loading")
            model=load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)
            #Loads model.pkl (pre-trained ML model).
            #Loads preprocessor.pkl (for transforming input data).
            print("After Loading")
            #model = RandomForestRegressor() (pre-trained model).
            #preprocessor = StandardScaler() (scales numerical inputs).
            scaled_data=preprocessor.transform(features)
            preds = model.predict(scaled_data)
            #Transforms features using preprocessor → Converts raw input into model-compatible format.
            #Makes predictions using model.predict(data_scaled).
            #eg-Input Data-{"gender": "male", "reading_score": 80, "writing_score": 85}
            #Transformed Data-[[0, 80, 85]]  # Normalized values
            #[78]  # Predicted math score
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        



#CustomData Class (Handles User Input)

class CustomData:
    def __init__(self, gender: str,
        race_ethnicity: str,
        parental_level_of_education:str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        #Stores form inputs (user-submitted data from home.html).
        #Example of Input Data:
        #CustomData(gender="male", race_ethnicity="group A",
        #parental_level_of_education="high school",
        #lunch="standard", test_preparation_course="none",
        #reading_score=78, writing_score=80)
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        #Saves form input as class attributes for easy access.
        #eg-self.gender = "male",self.reading_score = 78

    def convert_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        #Creates a dictionary of form data.
        #Converts it into a Pandas DataFrame for ML model input.
        #Dictionary format input-
        #{"gender": ["male"],"reading_score": [78],"writing_score": [80]}
        #Dictionary input converted to a Pandas Dataframe-
        #  gender	reading_score	writing_score
        #   male       78               80

        except Exception as e:
            raise CustomException(e,sys)
        


#1.User fills out the form in home.html-Data is sent to /predictdata (Flask route).
#2.CustomData stores user input-Converts data into a Pandas DataFrame.
#3.PredictPipeline.predict(features) runs
#- Loads the model and preprocessor.
#- Transforms the input using preprocessor.
#- Predicts the student’s math score.
#4.Prediction is sent back to home.html-The predicted score is displayed on the webpage!






