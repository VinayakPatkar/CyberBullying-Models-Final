from time import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib
import os
import datetime

def pipeline(learner_list, X_train, y_train, X_test, y_test): 
    size = len(y_train)
    results = {}
    final_results = []
    model_store_path = 'C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/model_store'

    for learner in learner_list:
        results['Algorithm'] = learner.__class__.__name__
        start = time() 
        print("Training {}".format(learner.__class__.__name__))
        learner = learner.fit(X_train, y_train)
        end = time()
        results['Training Time'] = end - start
        start = time()
        predictions_test = learner.predict(X_test)
        predictions_train = learner.predict(X_train)
        end = time()
        results['Prediction Time'] = end - start
        results['Accuracy: Test'] = accuracy_score(y_test, predictions_test)
        results['Accuracy: Train'] = accuracy_score(y_train, predictions_train)
        results['F1 Score: Test'] = f1_score(y_test, predictions_test)
        results['F1 Score: Train'] = f1_score(y_train, predictions_train)
        results['Precision: Test'] = precision_score(y_test, predictions_test)
        results['Precision: Train'] = precision_score(y_train, predictions_train)
        results['Recall: Test'] = recall_score(y_test, predictions_test)
        results['Recall: Train'] = recall_score(y_train, predictions_train)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{learner.__class__.__name__}_{timestamp}_model.pkl"
        model_path = os.path.join(model_store_path, model_filename)
        joblib.dump(learner, model_path)
        results['Model Path'] = model_path
        print("Training {} finished in {:.2f} sec".format(learner.__class__.__name__, results['Training Time']))
        print('Model saved to: {}'.format(model_path))
        print('----------------------------------------------------')
        
        final_results.append(results.copy())
    return final_results
