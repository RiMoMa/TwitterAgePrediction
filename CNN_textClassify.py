from multiprocessing import  Pool
import pandas as pd
import  numpy as np
import re
import os.path
#import functions to clean data
import unidecode
from nltk.stem import *
from nltk.stem.porter import *
from cleanData import remove_punctuation, remove_users,remove_stopwords,Build_dictionary,separate_users,separate_hashtags,remove_hashtags,lemmatization, Separate_hashtag_words

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
data['copy']=data['text']
stemmer = PorterStemmer() #stemmer class
data['text'] = data.text.apply(lambda x: stemmer.stem(x)) #to remove s (to singular)
data['text'] = data.text.apply(lambda x: unidecode.unidecode(x)) #to covert all in ascii
data['text'] = data.text.apply(lambda x: remove_users(x)) # to remove users
data['text'] = data.text.apply(lambda x: remove_hashtags(x)) #to remove hastags
data['text'] = data.text.apply(lambda x: remove_stopwords(x)) # remove stop words, and spanish
data["text"] = data["text"].str.replace(r'https?://\S+|www\.\S+', ' ').str.strip() # to delete all URLS
data["text"] = data["text"].str.replace(r'\n', ' ').str.strip() # to delete all '\n'
data['text'] = data['text'].apply(lambda x:remove_punctuation(x)) #delete punctuation


### FUNCTION TO LEMMAZATION THE DATA IN PARALLEL
def lemmaParallel(data):
    data['textLemma'] = data['text'].apply(lambda x:lemmatization(x))
    return data

def parallelize_dataframe(df, func, n_cores=16):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

### IF DOES NOT EXIST THE LEMMA DATA THEN EXCUTE THE FUNCTION IF EXIST OPEN THE DATA
if not os.path.exists(cleanDataPath):
    data = parallelize_dataframe(data,lemmaParallel)
    data.to_pickle(cleanDataPath)
else:
    data = pd.read_pickle(cleanDataPath)


### To separate hasgtags word, remove punctuation and low
data['Hashtags'] =data.Hashtags.apply(lambda x:Separate_hashtag_words(x)) #delete punctuation
data['Hashtags'] = data['Hashtags'].apply(lambda x:remove_punctuation(x))
data['Hashtags']= data['Hashtags'].apply(lambda x: x.lower())
data['users'] = data['users'].apply(lambda x:remove_punctuation(x))

## To remove empty text
data.drop(data[data['textLemma'] == ''].index)

data['text'] = data.users+' '+data.textLemma+' '+data.Hashtags


#SPliting train and test

from sklearn.model_selection import train_test_split
X_train, X_testVal, y_train, y_testVal = train_test_split(
    data['text'] ,data.age_range, test_size=0.35, random_state=42)

X_Val, X_Test, y_Val, y_test = train_test_split(
    X_testVal,y_testVal, test_size=0.70, random_state=42)


y_test2=y_test



from tensorflow.keras.utils import to_categorical
y_train =to_categorical(y_train)
y_test = to_categorical(y_test)
y_Val = to_categorical(y_Val)
## TOKENIZATION

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=vocab_size,oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
Training_data = tokenizer.texts_to_sequences(X_train)
Val_data = tokenizer.texts_to_sequences(X_Val)
Testing_data = tokenizer.texts_to_sequences(X_Test)
maxlen = 0
for i in range(len(Training_data)):
  if len(Training_data[i]) > maxlen:
    maxlen = len(Training_data[i])

x_train = tf.keras.preprocessing.sequence.pad_sequences(Training_data, maxlen=maxlen, padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(Testing_data, maxlen=maxlen, padding='post')
x_Val = tf.keras.preprocessing.sequence.pad_sequences(Val_data, maxlen=maxlen, padding='post')


## generate sample
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=27)
#x_train, y_train = sm.fit_resample(x_train, y_train)




tf.keras.backend.clear_session()


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=maxlen),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.GlobalAveragePooling1D(),
   # tf.keras.layers.Dense(40, activation='relu'),
   # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])




model.compile(loss='CategoricalCrossentropy',optimizer='adam',metrics=['categorical_accuracy'])

model.summary()
filepath = 'model.h5'
mc = tf.keras.callbacks.ModelCheckpoint(filepath, verbose=0, save_weights_only=True,
                                        monitor='val_loss', mode='auto', save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, mode='auto')

import timeit

start = timeit.default_timer()

history = model.fit(x_train, y_train, epochs=100, batch_size=160,
                    validation_data=(x_Val, y_Val), callbacks=[mc, es])


stop = timeit.default_timer()

print('Time: ', stop - start)


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
y_preds2=np.argmax(y_pred,axis=1)
a=np.array(y_preds2)
b=y_test2.to_numpy()
confusion_matrix(b,a)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
y_pred = model.predict(x_test)
y_preds2=np.argmax(y_pred,axis=1)
a=np.array(y_preds2)
b=y_test2.to_numpy()
print(confusion_matrix(b,a))
print('F-score:  ',f1_score(b, a, average='macro'))
from sklearn.metrics import accuracy_score
print('Accuaracy: ',accuracy_score(b, a))

