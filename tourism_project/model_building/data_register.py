from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

#1. First create a repo id for hugging face as we are going to load files related to tourism
repo_id = "SriniGS/tourism-package-prediction-v2"

#2. Define the repo type as dataset
repo_type = "dataset"

# 3. Initialize the API by using the HF token. This will establish the connection with HF
api = HfApi(token=os.getenv("HF_TOKEN"))

# 4. Check, if the space already exist on Hugging Face.
# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# 5. Upload the files to the repo =dataset
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
