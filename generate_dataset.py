import pandas as pd
import numpy as np

np.random.seed(42)

n_rows = 1000

# Categorical features
genders = ['male', 'female']
edu_levels = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
prep_courses = ['none', 'completed']
study_times = ['0-3 hrs', '3-5 hrs', '5-10 hrs', '> 10 hrs']

data = {
    'gender': np.random.choice(genders, n_rows),
    'parental_level_of_education': np.random.choice(edu_levels, n_rows, p=[0.15, 0.2, 0.25, 0.2, 0.15, 0.05]),
    'test_preparation_course': np.random.choice(prep_courses, n_rows, p=[0.65, 0.35]),
    'study_time': np.random.choice(study_times, n_rows, p=[0.2, 0.35, 0.3, 0.15]),
}

df = pd.DataFrame(data)

# Generate 5 pre-exam scores (correlated with each other via a hidden "aptitude" variable)
aptitude = np.random.normal(65, 12, n_rows)

df['math_pre_score'] = np.clip(np.random.normal(aptitude, 8), 0, 100).astype(int)
df['biology_pre_score'] = np.clip(np.random.normal(aptitude + 2, 8), 0, 100).astype(int)
df['chemistry_pre_score'] = np.clip(np.random.normal(aptitude - 2, 8), 0, 100).astype(int)
df['physics_pre_score'] = np.clip(np.random.normal(aptitude - 4, 10), 0, 100).astype(int)
df['english_pre_score'] = np.clip(np.random.normal(aptitude + 5, 7), 0, 100).astype(int)

# Target variable: final_score
# Dependent mostly on pre-exam scores, but also study time and preparation
avg_pre = (df['math_pre_score'] + df['biology_pre_score'] + df['chemistry_pre_score'] + df['physics_pre_score'] + df['english_pre_score']) / 5

study_boost = {'0-3 hrs': -4, '3-5 hrs': 0, '5-10 hrs': 5, '> 10 hrs': 9}
prep_boost = {'none': -2, 'completed': 6}
edu_boost = {"some high school": -2, "high school": -1, "some college": 0, "associate's degree": 1, "bachelor's degree": 2, "master's degree": 3}

boosts = (
    df['study_time'].map(study_boost) + 
    df['test_preparation_course'].map(prep_boost) + 
    df['parental_level_of_education'].map(edu_boost)
)

final_score_raw = avg_pre * 0.85 + boosts + np.random.normal(5, 3, n_rows)
df['final_score'] = np.clip(final_score_raw, 0, 100).astype(int)

# Reorder columns
df = df[['gender', 'parental_level_of_education', 'test_preparation_course', 'study_time', 
         'math_pre_score', 'biology_pre_score', 'chemistry_pre_score', 'physics_pre_score', 'english_pre_score', 'final_score']]

# Save to the specific location
output_path = r'C:\Users\bhars\.gemini\antigravity\scratch\student_performance_prediction\notebook\data\stud.csv'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Generated {n_rows} rows of synthetic dataset at {output_path}")
print("Head:")
print(df.head())
