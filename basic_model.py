from utils.basic_model_pipeline import pipeline
from utils.data_loading import load_data
from preprocesser.clean_text_new import preprocess,text_preprocessing_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas as pd
data = load_data()
data["text"] = data["text"].apply(lambda x : preprocess(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
'''gpt_data["text"] = gpt_data["text"].apply(lambda x : preprocess(x))
gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]'''
data["text"] = data["text"].apply(lambda x : text_preprocessing_pipeline(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
'''gpt_data["text"] = gpt_data["text"].apply(lambda x : text_preprocessing_pipeline(x))
gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]'''
print(type(data['label'][0]))
X_train, X_test, y_train, y_test = train_test_split(data['text'], 
                                                    data['label'], 
                                                    random_state=42)

print('Number of rows in the total set: {}'.format(data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
count_vector = CountVectorizer(stop_words = 'english', lowercase = True)
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
models = [MultinomialNB(), DecisionTreeClassifier(), LinearSVC(), AdaBoostClassifier(), 
          RandomForestClassifier(), BaggingClassifier(),
         LogisticRegression(), SGDClassifier(), KNeighborsClassifier()]
re = pipeline(models, training_data, y_train, testing_data, y_test)
results = pd.DataFrame(re)
results = results.reindex(columns = ['Algorithm', 'Accuracy: Test', 'Precision: Test', 'Recall: Test', 'F1 Score: Test', 'Prediction Time',
                          'Accuracy: Train', 'Precision: Train', 'Recall: Train', 'F1 Score: Train', 'Training Time'])
results.to_csv('results/results.csv', index=False)
results = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/results/results.csv")
best_acc = results[results['Accuracy: Test'] == results['Accuracy: Test'].max()]
best_f1 = results[results['F1 Score: Test'] == results['F1 Score: Test'].max()]
best_precision = results[results['Precision: Test'] == results['Precision: Test'].max()]
best_recall = results[results['Recall: Test'] == results['Recall: Test'].max()]
print("Best Accuracy : ",best_acc);
print("Best F1 Score : ",best_f1);
print("Best Precision : ",best_precision);
print("Best Recall : ",best_recall);
best_train_time = results[results['Training Time'] == results['Training Time'].min()]
worst_train_time = results[results['Training Time'] == results['Training Time'].max()]
best_prediction_time = results[results['Prediction Time'] == results['Prediction Time'].min()]
worst_prediction_time = results[results['Prediction Time'] == results['Prediction Time'].max()]
print("Best Train Time : ",best_train_time);
print("Worst Train Time : ",worst_train_time);
print("Best Prediction Time : ",best_prediction_time);
print("Worst Prediction Time : ",worst_prediction_time);