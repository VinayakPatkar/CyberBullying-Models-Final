from preprocesser.clean_text_new import preprocess,text_preprocessing_pipeline
import pandas as pd
from scraped_preprocessing.language_proc import filter_english_comments
from scraped_preprocessing.mapping import returnlabels
#filter_english_comments()
data = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/threeaggdata/cyberbullying_tweets.csv")
data = data.rename(columns = {"tweet_text" : "text","cyberbullying_type" : "label"})
#data = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/data_store/english_comments.csv");
data = data.rename(columns = {"tweet_text" : "text","cyberbullying_type" : "label"})
#labels = returnLabels()
#print("Before:",labels)
data["text"] = data["text"].apply(lambda x : preprocess(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
data["text"] = data["text"].apply(lambda x : text_preprocessing_pipeline(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
data['label'] = data['label'].replace({'cyberbullying': 1, 'not_cyberbullying': 0,'gender' : 1,'religion':1,'other_cyberbullying' : 1,'age' : 1,'ethnicity' : 1})
#labels = returnLabels()
#print(data)
#print("After : ",labels)
print(data)