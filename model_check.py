import os
import joblib
import time

def load_models(model_directory):
    model_files = os.listdir(model_directory)
    models = {}
    for file in model_files:
        if file.endswith(".pkl"):
            model_name = file.split("_")[0]  
            model_path = os.path.join(model_directory, file)
            if model_name not in models:
                models[model_name] = []
            models[model_name].append((file, joblib.load(model_path)))
    return models
model_store_path = 'C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/model_store'
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
while True:
    clear_console()  
    loaded_models = load_models(model_store_path)
    for model_name, model_list in loaded_models.items():
        print(f"Models for {model_name}:")
        for model_file, model in model_list:
            print(f"- {model_file}")
    time.sleep(60)  
