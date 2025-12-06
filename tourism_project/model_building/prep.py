# 1. Prepare for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants and placeholders
HF_USER_ID = "SriniGS"
DATASET_PATH = "hf://datasets/SriniGS/tourism-package-prediction-v2/tourism.csv"
REPO_ID = "SriniGS/tourism-package-prediction-v2"
TARGET_COL = 'ProdTaken'
print("::: Data Preparation ::: Step 1 ::: Prepare for data manipulation.")

# Ensure the HF_TOKEN environment variable is set.
api = HfApi(token=os.getenv("HF_TOKEN"))
print("::: Data Preparation ::: Step 2 ::: Hugging Face API initialized.")

# 3. Read the csv file
try:
    df = pd.read_csv(DATASET_PATH)
    print("::: Data Preparation ::: Step 3 ::: Dataset loaded successfully.")
except Exception as e:
    print(f"::: Data Preparation ::: Step 3 ::: Error loading dataset: {e}")
    # For local testing if HF access is an issue
    # df = pd.read_csv("tourism.csv")

# 4. Drop the unique identifier
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)
print("::: Data Preparation ::: Step 4 ::: Dropped identifier columns.")

# 5. Seperate the categorical and numerical variables of Indepenent variables.
# This list is used for documentation/later model building script
numerical_cols = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
categorical_cols = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
]
print("::: Data Preparation ::: Step 5 ::: Seperated numerical and categorical variables.")
print(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features.")


# 6. Seperate the target variable and independent variables
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print("::: Data Preparation ::: Step 6 ::: Seperated target variable and independent variables.")

# 7. Split the dataset into train test using train test split function
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class ratio
)
print("::: Data Preparation ::: Step 7 ::: Split the dataset into train and test sets.")

# 8. train and test data - load this data into respective csv files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("::: Data Preparation ::: Step 8 ::: Train/test data split and saved locally as CSVs.")

# 9. upload these csv into hugging face respository
files_to_upload = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files_to_upload:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            # UPLOADING FILENAME ONLY TO REPO ROOT
            path_in_repo=file_path.split("/")[-1],
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"::: Data Preparation ::: Step 9 ::: Successfully uploaded: {file_path}")
    except Exception as e:
        print(f"Failed to upload {file_path}: {e}")
        print("NOTE: Ensure you have write access to the repository and the HF_TOKEN is valid.")

print("::: Data Preparation ::: Step 10 ::: Data preparation script completed.")
