
import pandas as pd
import  numpy as np
import re
import os.path
#import functions to clean data
import unidecode
from nltk.stem import *
from nltk.stem.porter import *
from cleanData import remove_punctuation, remove_at_symbol,Separate_at_words,remove_users,remove_stopwords,remove_hash_symbol,count_punctuation,separate_users,separate_hashtags,remove_hashtags,lemmatization, Separate_hashtag_words,separate_urls
###OPTIONS

vocab_size=4000
cleanDataPath = 'DataClean.pkl' # Path for the postprocessing data

#load Original Data
data = pd.read_json('data/clasificador/clasificador.json') #To open json
label_text = data['age_range'].unique()#To determine ranges or classes
label_text.sort() # organize the age ranges
for n in range(len(label_text)):
    data['age_range']=data.age_range.replace(label_text[n], n) #to remplace the str class with a numerical class

label_n = np.array(range(len(label_text)))#To define numerical classes associated with age-ranges

# to divide the data in user, hasgtags and text
data['users']=data.text.apply(lambda x:separate_users(x))
data['Hashtags']=data.text.apply(lambda x:separate_hashtags(x))
data['Urls']=data.text.apply(lambda x:separate_urls(x))

data['copy']=data['text']
data['text'] = data.text.apply(lambda x: remove_users(x)) # to remove users
data['text'] = data.text.apply(lambda x: remove_hashtags(x)) #to remove hastags
data["text"] = data["text"].str.replace(r'https?://\S+|www\.\S+', ' ').str.strip() # to delete all URLS

data['HashtagsW'] =data.Hashtags.apply(lambda x:Separate_hashtag_words(x)) #delete punctuation
data['HashtagsW'] =data.HashtagsW.apply(lambda x:remove_hash_symbol(x)) #delete punctuation
data['usersW'] =data.users.apply(lambda x:Separate_at_words(x)) #delete punctuation
data['usersW'] =data.usersW.apply(lambda x:remove_at_symbol(x)) #delete punctuation
#data['textW'] =data.text.apply(lambda x:Separate_at_words(x)) #delete punctuation
#data['textW'] =data.textW.apply(lambda x:remove_at_symbol(x)) #delete punctuation
 #delete punctuation

import re
string = "Not mAnY Capital Letters"
len(re.findall(r'[A-Z]',string))
Features = pd.DataFrame()

######### Features For Hashtags
Features['NHashTags'] = data.Hashtags.apply(lambda x:len(x.split()) ) #number of hashtags
Features['NHashTagsUpperLetters'] = data.Hashtags.apply(lambda x:len(re.findall(r'[A-Z]',x))) #Number of upper letter in hashtags
Features['NHashTagsLowLetters'] = data.Hashtags.apply(lambda x:len(re.findall(r'[a-z]',x)))#Number of lower letter in hashtags
Features['MaxLenHashTagsWords'] = data.Hashtags.apply(lambda x: len(max(x.split(), key=len))-1 if bool(x) else 0 ) #max len hashtag
Features['MinLenHashTagsWords'] = data.Hashtags.apply(lambda x: len(min(x.split(), key=len))-1 if bool(x) else 0 )#min len hashtah
Features['NWordsTags'] = data.HashtagsW.apply(lambda x:len(x.split()) ) #number of words in the hashtags
Features['MaxLenHashTagsWords_se'] = data.HashtagsW.apply(lambda x: len(max(x.split(), key=len))-1 if bool(x) else 0 ) #max len hashtag word
Features['MinLenHashTagsWords_se'] = data.HashtagsW.apply(lambda x: len(min(x.split(), key=len))-1 if bool(x) else 0 )#min len hashtah word
Features['NHashTagsnumbers'] = data.Hashtags.apply(lambda x:len(re.findall(r'[0-9]',x)))#Number of numbers
Features['NHashTagsSymbols'] = data.HashtagsW.apply(lambda x:count_punctuation(x))#Number of symbols hashtags
######

####Features For Users
Features['NUsers'] = data.users.apply(lambda x:len(x.split()) ) #number of users
Features['NUsersUpperLetters'] = data.users.apply(lambda x:len(re.findall(r'[A-Z]',x))) #Number of upper letter in hashtags
Features['NUsersLowLetters'] = data.users.apply(lambda x:len(re.findall(r'[a-z]',x)))#Number of lower letter in hashtags
Features['MaxUsersWords'] = data.users.apply(lambda x: len(max(x.split(), key=len))-1 if bool(x) else 0 ) #max len hashtag
Features['MinLenUsersWords'] = data.users.apply(lambda x: len(min(x.split(), key=len))-1 if bool(x) else 0 )#min len hashtah
Features['NWordsUsers'] = data.usersW.apply(lambda x:len(x.split()) ) #number of words in the hashtags
Features['MaxLenUsers_se'] = data.usersW.apply(lambda x: len(max(x.split(), key=len))-1 if bool(x) else 0 ) #max len hashtag word
Features['MinLenUsers_se'] = data.usersW.apply(lambda x: len(min(x.split(), key=len))-1 if bool(x) else 0 )#min len hashtah word
Features['NUsersnumbers'] = data.users.apply(lambda x:len(re.findall(r'[0-9]',x)))#Number of numbers used in users
Features['NUsersSymbols'] = data.usersW.apply(lambda x:count_punctuation(x))#Number of symbols users delete
######


####Features For texts
Features['NUtexts'] = data.text.apply(lambda x:len(x.split()) ) #number of users
Features['NUtextsUpperLetters'] = data.text.apply(lambda x:len(re.findall(r'[A-Z]',x))) #Number of upper letter in hashtags
Features['NUtextsLowLetters'] = data.text.apply(lambda x:len(re.findall(r'[a-z]',x)))#Number of lower letter in hashtags
Features['MaxtextsWords'] = data.text.apply(lambda x: len(max(x.split(), key=len))-1 if bool(x) else 0 ) #max len hashtag
Features['MinLentextsWords'] = data.text.apply(lambda x: len(min(x.split(), key=len))-1 if bool(x) else 0 )#min len hashtah
Features['NTextssnumbers'] = data.text.apply(lambda x:len(re.findall(r'[0-9]',x)))#Number of numbers used in text
Features['NTextsSymbols'] = data.text.apply(lambda x:count_punctuation(x))#Number of symbols text delete
######
Features['UrlExist'] = data.Urls.apply(lambda x: 1 if bool(x) else 0 )#min len hashtah



S=np.array(Features)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    S ,data.age_range, test_size=0.35, random_state=42)


from sklearn.ensemble import AdaBoostClassifier

#Your statements here

import timeit

start = timeit.default_timer()

clf = AdaBoostClassifier(n_estimators=300, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


stop = timeit.default_timer()

print('Time: ', stop - start)

b=y_test
a=y_pred
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


print('confusion matrix')
print(confusion_matrix(b,a))
print('accuaracy: ',accuracy_score(b, a))
print('F-score:  ',f1_score(b, a, average='macro'))

clf = AdaBoostClassifier(n_estimators=300, random_state=0)
clf.fit(X_train, y_train<3)
y_pred = clf.predict(X_test)
b=y_test<3
a=y_pred


print('confusion matrix')
print(confusion_matrix(b,a))

print('Two Classes , accuaracy: ',accuracy_score(b, a))
print('Two Classes,F-score:  ',f1_score(b, a, average='macro'))

## generate sample
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=27)
X_train, y_train = sm.fit_resample(X_train, y_train)

clf = AdaBoostClassifier(n_estimators=300, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
b=y_test
a=y_pred


print('confusion matrix')
print(confusion_matrix(b,a))

print('Augmented data , accuaracy: ',accuracy_score(b, a))
print('Augmented data ,F-score:  ',f1_score(b, a, average='macro'))

clf = AdaBoostClassifier(n_estimators=300, random_state=0)
clf.fit(X_train, y_train<3)
y_pred = clf.predict(X_test)
b=y_test<3
a=y_pred

print('confusion matrix')
print(confusion_matrix(b,a))

print('Two clases Augmented data , accuaracy: ',accuracy_score(b, a))
print('Two clases Augmented data ,F-score:  ',f1_score(b, a, average='macro'))

from sklearn.manifold import TSNE
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
feat_cols = [ 'pixel'+str(i) for i in range(X_test.shape[1]) ]
df = pd.DataFrame(X_test,columns=feat_cols)
df['y']=y_test>3


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_test)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.3
)
plt.show()
print('end')