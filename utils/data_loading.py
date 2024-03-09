import pandas as pd
def load_data():
    data_CCD = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/threeaggdata/CyberBullying Comments Dataset.csv")
    data_c_t = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/threeaggdata/threeaggtrain.csv")
    data_c_d = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/threeaggdata/threeaggtest.csv")
    data_cc_ct = pd.read_csv("C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/threeaggdata/cyberbullying_tweets.csv")
    data_c_t = data_c_t.rename(columns = {"Well said sonu..you have courage to stand against dadagiri of Muslims" : "text","OAG" : "label"})
    data_c_t = data_c_t.drop(["facebook_corpus_msr_1723796"],axis = 1)
    data_c_d = data_c_d.rename(columns = {"The quality of re made now makes me think it is something to be bought from fish market" : "text","CAG" : "label"})
    data_c_d = data_c_d.drop(["facebook_corpus_msr_451811"],axis = 1)
    data_cc_ct = data_cc_ct.rename(columns = {"tweet_text" : "text","cyberbullying_type" : "label"})
    data_CCD = data_CCD.rename(columns = {"Text" : "text","CB_Label" : "label"})
    data_c_t['label'] = data_c_t['label'].replace({'OAG': 1, 'CAG': 1, 'NAG': 0})
    data_c_d['label'] = data_c_d['label'].replace({'OAG': 1, 'CAG': 1, 'NAG': 0})
    data_cc_ct['label'] = data_cc_ct['label'].replace({'cyberbullying': 1, 'not_cyberbullying': 0,'gender' : 1,'religion':1,'other_cyberbullying' : 1,'age' : 1,'ethnicity' : 1})
    data_joined = pd.concat([data_c_t,data_c_d,data_cc_ct,data_CCD],ignore_index=True)
    return data_joined