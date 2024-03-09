from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from time import time
import joblib
import os

def param_tuning(clf, param_dict, X_train, y_train, X_test, y_test):
    scorer = make_scorer(f1_score)
    grid_obj = GridSearchCV(estimator=clf,
                            param_grid=param_dict,
                            scoring=scorer,
                            cv=5)

    grid_fit = grid_obj.fit(X_train, y_train)
    best_clf = grid_fit.best_estimator_
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    print(clf.__class__.__name__)
    print("\nOptimized Model\n------")
    print("Best Parameters: {}".format(grid_fit.best_params_))
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("F1-score: {:.4f}".format(f1_score(y_test, best_predictions)))
    print("Precision: {:.4f}".format(precision_score(y_test, best_predictions)))
    print("Recall: {:.4f}".format(recall_score(y_test, best_predictions)))

    # Save the model
    model_name = clf.__class__.__name__
    timestamp = int(time())
    params_str = '_'.join([f"{key}_{value}" for key, value in grid_fit.best_params_.items()])
    model_filename = f"{model_name}_{timestamp}_{params_str}.pkl"
    model_path = os.path.join("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/model_store", model_filename)
    joblib.dump(best_clf, model_path)
    print('Model saved to:', model_path)
    
    return best_clf, grid_fit.best_params_, model_path
