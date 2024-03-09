from utils.hyperparam_pipeline import param_tuning
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
data = load_data()
data["text"] = data["text"].apply(lambda x : preprocess(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
'''gpt_data["text"] = gpt_data["text"].apply(lambda x : preprocess(x))
gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]'''
data["text"] = data["text"].apply(lambda x : text_preprocessing_pipeline(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
'''gpt_data["text"] = gpt_data["text"].apply(lambda x : text_preprocessing_pipeline(x))
gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]'''
X_train, X_test, y_train, y_test = train_test_split(data['text'], 
                                                    data['label'], 
                                                    random_state=42)

print('Number of rows in the total set: {}'.format(data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
count_vector = CountVectorizer(stop_words = 'english', lowercase = True)
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
param_grid = {
    'alpha' : [0.095, 0.0002, 0.0003],
    'max_iter' : [2500, 3000, 4000]
}

clf_sgd = SGDClassifier()

param_tuning(clf_sgd, param_grid, training_data, y_train, testing_data, y_test)

param_grid = {
    'C': [1, 1.2, 1.3, 1.4]
}

clf_lr = LogisticRegression()

param_tuning(clf_lr, param_grid, training_data, y_train, testing_data, y_test)

param_grid = {
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 5, 8]
}

clf_dt = DecisionTreeClassifier()

param_tuning(clf_dt, param_grid, training_data, y_train, testing_data, y_test)
param_grid = {
    'n_estimators': [50,150],
    'min_samples_leaf': [1, 5],
    'min_samples_split': [2, 5]
}

clf_rf = RandomForestClassifier()

param_tuning(clf_rf, param_grid, training_data, y_train, testing_data, y_test)

param_grid = {
    'C': [0.25, 0.5, 0.75, 1, 1.2]
}

clf_linsvc = LinearSVC()

param_tuning(clf_linsvc, param_grid, training_data, y_train, testing_data, y_test)