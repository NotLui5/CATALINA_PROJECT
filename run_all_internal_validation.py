from pathlib import Path
import subprocess
import sys


MODELS = [
    "CatBoost",
    "GaussianNB",
    "KNN",
    "LightGBM",
    "RandomForest",
    "SVC",
    "XGBoost",
    "LogisticRegression"
]

FEATURED_OPTIONS = [
    "lasso",
    "elasticnet"
]

MODEL_TYPES = [
    "tuned",
    "base"
]

SCRIPT_PATH = "step5_internal_validation.py"

time = ('15') ########## Varies according to followup in years.
location_data_IV = ('./database/imputation_mice/')
path_to_models_prefix = "./results/models/"

for model in MODELS:
    for featured in FEATURED_OPTIONS:
        for model_type in MODEL_TYPES:

            if model_type == "tuned":
                if int(time) >= 50:
                    path_models_sufix = (
                        f"{model}_tuned_{featured}_tuned_10_imputations.pkl"
                    )
                else:
                    path_models_sufix = (
                        f"{model}_tuned_{featured}{time}ly_tuned_10_imputations.pkl"
                    )
            else:
                if int(time) >= 50:
                    path_models_sufix = (
                        f"{model}_{featured}_base_10_imputations.pkl"
                    )
                else:
                    path_models_sufix = (
                        f"{model}_{featured}{time}ly_base_10_imputations.pkl"
                    )
            model_selected_path = path_to_models_prefix + path_models_sufix
            model_path = Path(model_selected_path)

            print("\n" + "=" * 80)
            print(f"MODEL: {model}")
            print(f"FEATURED: {featured}")
            print(f"MODEL TYPE: {model_type}")
            print(f"PATH: {model_selected_path}")
            print(f"TIME-FOLLOWUP: {time} last years")
            print("=" * 80)

            # Skip combinations whose model file does not exist
            if not model_path.exists():
                print(f"Model file not found. Skipping: {model_selected_path}")
                continue

            command = [
                sys.executable,
                SCRIPT_PATH,
                "--model",
                model,
                "--featured",
                featured,
                "--model-type",
                model_type,
                "--model-selected-path",
                model_selected_path,
                "--path-data-iv",
                location_data_IV,
                "--time-followup", 
                time
            ]

            try:
                subprocess.run(
                    command,
                    check=True
                )

                print(
                    f"Completed: {model} | "
                    f"{featured} | {model_type}"
                )

            except subprocess.CalledProcessError as error:
                print(
                    f"Error processing: {model} | "
                    f"{featured} | {model_type}"
                )
                print(f"Return code: {error.returncode}")