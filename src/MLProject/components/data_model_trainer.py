import os
import sys
from dataclasses import dataclass

#Ensemble Regressor algorithms
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from xgboost import XGBRegressor

#Generic ML Regressor 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#Performance metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error

from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
from src.MLProject.utils import train_and_evaluate_models
from src.MLProject.utils import save_obj

import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

#Goal-Similar to DataTransformer/DataIngestion
#where we make the Config class to form the directory to store the pikle file 
# directory which will store the trained models
#Second use the ModelTrainer class to call the Transformed train and test datasets
# and train the model andf then save the pickle file of the model

@dataclass

class ModTrainConfig:
    trained_mod_filepath=os.path.join("artifact","model.pkl")
    #The best model with best performane will be stored in the pickle file

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModTrainConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    def initiate_mod_train(self,train_array,test_array):
        # 1.will input the train+test array
        # 2.Train the model+Use Hyperparam.Tuning
        # 3.Pick the best performed model and return its object(functionality)
        #to be stored in a pickle file

        try:
            logging.info("Split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            
            #Declaring the list of models

            logging.info(f"Declaring the models and params")

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
                }
            
            #Enlisting list parameter values for each model mentioned

            params={
                "Random Forest":{
                     #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                     #"max_features":['sqrt', 'log2'],
                    'n_estimators':[8, 16, 32, 64, 128, 256]},

                "Decision Tree":{
                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                     #'splitter':['best', 'random']},
                     #"max_features":['sqrt', 'log2']},

                "Linear Regression":{},

                "Gradient Boosting":{
                     #'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate':[0.1,0.01,0.05,0.0001],
                     #'criterion':['squared_error', 'friedman_mse'],
                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators':[8, 16, 32, 64, 128, 256]},
                     #'max_features':['sqrt', 'log2']},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]},

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                     #'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]}
                    
                    }
            
            print(models.keys(),params.keys())
            logging.info(f"{models.keys()},{params.keys()}")
            #The evaluate_models() from utils file will input all these models,perform 
            #Hyperparam tuning and return the best model among them with the model score etc.

            #Model report that will be returned will be of dictionary type

            logging.info(f"Generating model report")

            report=train_and_evaluate_models(models, params, X_train, y_train, X_test, y_test)
            #Format- Model Name:Model Score

            #From this dictionary,fetch the best model score by sorting

            logging.info(f"Fetching best model score") 

            # Get the best model (highest R² score)
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]
            best_model = models[best_model_name]  # Retrieve the model object


            # Print results
            print("Final Model Performance Report (R² Scores):")
            print(report)
    
            print(f"\n🏆 Best Model: {best_model_name} with R² Score: {best_model_score}")

            model_names=list(params.keys())
            actual_model=""

            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model

            best_params=params[actual_model] #parameters of actual model

            """

            #Now we use mlflow pipeline to track every experiment(every running of the entire project with logs and hyperparams)
            # This code is using MLflow to track and log machine learning experiments, 
            # and it is integrated with DagsHub for model tracking.


            mlflow.set_registry_uri("https://dagshub.com/sagnikbhatt26/my-first-repo.mlflow")
            #This line tells MLflow to use DagsHub as the registry for storing and managing models.
            #DagsHub is like GitHub for machine learning, where you can store models, datasets, and experiments.

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            #mlflow.get_tracking_uri() returns the tracking URL MLflow is using.
            #urlparse(...).scheme extracts the type of storage (e.g., http, https, or file).
            #This is useful later when deciding how to save the model.



            # mlflow

            #This starts an MLflow run, which is like pressing "record" to track the training process.
            #Everything logged inside this "with" block will be stored in MLflow.
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)
                #The best_model is used to make predictions on X_test.
                #The results are stored in predicted_qualities.

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)
                #This calls a function eval_metrics() (defined elsewhere) to calculate:
                #RMSE (Root Mean Square Error) → Measures how far predictions are from actual values.
                #MAE (Mean Absolute Error) → Measures the average error in predictions.
                #R² (R-squared score) → Measures how well the model explains the data.

                mlflow.log_params(best_params)
                #Logs the hyperparameters of best_model in MLflow.
                #Helps track which parameters were used to train this model.

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                #These lines save the evaluation metrics in MLflow.
                #This helps compare different models later.


                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    #If the tracking URI is not a local file, then we register the model online.

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                    #If MLflow is not using a local file system, we register the model with a name (actual_model).
                    #This allows us to track different versions of the model in MLflow.


                else:
                    mlflow.sklearn.log_model(best_model, "model")
                    #If MLflow is using a local file system, we just log the model without registering it."
            """







            

            #Now we prepare a threshold for the best models
            #r_2 score should be above 0.6 cutoff

            if (best_model_score<0.6):
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")


            #Send the model trainer file path and object (functionality) to utils 
            #save_obj() function to form and get back the model trainer pickle file

            save_obj(
                file_path=self.model_trainer_config.trained_mod_filepath,
                obj=best_model)
            

            #Now make the predictions on X_test data w.r.t. the best model found

            pred=best_model.predict(X_test)

            r_2=r2_score(y_test,pred)

            return r_2
        
        except Exception as e:
            raise CustomException(e,sys)







        





