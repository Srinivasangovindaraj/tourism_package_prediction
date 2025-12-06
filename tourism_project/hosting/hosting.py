from huggingface_hub import HfApi
import os
#Initialize HF token as environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USER_ID = "SriniGS" # HF_USER
MODEL_REPO_ID = f"SriniGS/tourism-package-prediction-v2"
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=MODEL_REPO_ID,                        # the target repo
    repo_type="space",                            # dataset, model, or space
    path_in_repo="",                              # optional: subfolder path inside the repo
)
