import sys
import os
import pandas as pd
from src.utils import load_object
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 parental_level_of_education: str,
                 test_preparation_course: str,
                 study_time: str,
                 math_pre_score: int,
                 biology_pre_score: int,
                 chemistry_pre_score: int,
                 physics_pre_score: int,
                 english_pre_score: int):

        self.gender = gender
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.study_time = study_time
        self.math_pre_score = math_pre_score
        self.biology_pre_score = biology_pre_score
        self.chemistry_pre_score = chemistry_pre_score
        self.physics_pre_score = physics_pre_score
        self.english_pre_score = english_pre_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "parental_level_of_education": [self.parental_level_of_education],
                "test_preparation_course": [self.test_preparation_course],
                "study_time": [self.study_time],
                "math_pre_score": [self.math_pre_score],
                "biology_pre_score": [self.biology_pre_score],
                "chemistry_pre_score": [self.chemistry_pre_score],
                "physics_pre_score": [self.physics_pre_score],
                "english_pre_score": [self.english_pre_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
