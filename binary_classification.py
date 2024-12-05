import pandas as pd
import numpy as np
import cProfile
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
# Import data
df, dataf, engdf, engdataf = pd.read_csv("sanna_nyheter.csv"), pd.read_csv("falska_nyheter.csv"), pd.read_csv("fake_real_news_1.csv"), pd.read_csv("Fake_real_news_2.csv")

#Mergeing dataframes, concatinating(sammanfogar) objects within a list
frames, eng_frames= [df, dataf], [engdf, engdataf]
news_dataset, eng_news_dataset= pd.concat(frames), pd.concat(eng_frames)


#Remove unused columns from the datasets
news_dataset.drop([ 'ämne', 'titel'], axis=1, inplace=True)
eng_news_dataset.drop(['title', 'subject', 'date'], axis=1, inplace=True)
eng_news_dataset = eng_news_dataset.rename(columns={'target': 'label'})
print(eng_news_dataset)
#shorten down the eng_news_dataset, selecting 20000 random rows from the eng_news_dataset
eng_news_dataset= eng_news_dataset.sample(n=20000) 


# Mergeing dataframes
eng_swe_frames = [news_dataset, eng_news_dataset]
dataset = pd.concat(eng_swe_frames)

dataset= dataset.sample(frac=1)#Reshuffeling
dataset.reset_index(inplace=True)#Resetting index
dataset.drop(['index'], axis=1, inplace=True)
print(dataset)


print(dataset.isnull().sum())

# fill null values with empty strings
dataset = dataset.fillna('')
print(dataset.isnull().sum())
dataset['content']= dataset['text'] 
# Seperate data and label, X is the content and Y is the labels.
X=dataset.drop(columns='label', axis=1)
Y= dataset['label']


def stemming(content):
    # Remove urls, HTML tags, punctuation, digits, newline characters and stopwords
    stemmed_content = re.sub(r'https?://\S+|www\.\S+', '', content)
    stemmed_content = re.sub(r'<.*?>', '', content)
    stemmed_content = re.sub(r'[^\w\s]', '', content)
    stemmed_content = re.sub(r'\d', '', content)
    stemmed_content = re.sub(r'\n', '', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()   
    stemmed_content= ' '.join(stemmed_content)  
    return stemmed_content


# Apply stemming it will process a lot of words.
dataset['content'] = dataset['content'].apply(stemming)
print('Loading...')

print(dataset['content'])
# Store the content in X and label in y variabel
X=dataset['content'].values
Y=dataset['label'].values
print(dataset['content'])
print(Y.shape)
# convert textdata to numerical data. TfidfVectorizer() counts how many times a particually word is repeating in a document, text or paragraph. It defines a particually number to that.
# Justify the vectorizers dimensionality
vectorizer=TfidfVectorizer(max_features=1000)
vectorizer.fit(X)

X= vectorizer.transform(X)
print(X)

# text will be used for X data, remaining Y datam stratify = Y , random state 
# LogisticRegression uses the sigmoidfunction. The sigmoid funktion ensures that output are between 1 and 0. It uses linear combination of input and converts it into probabilities.

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=2)


model= LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuraccy = accuracy_score(X_train_prediction, Y_train)    

print('Accuracy score of the training data: ', training_data_accuraccy)

X_test_prediction = model.predict(X_test)
test_data_accuraccy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data: ', test_data_accuraccy)



DTC = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)

DTC.fit(X_train, Y_train)
pred_dtc= DTC.predict(X_test)


DTC.score(X_test, Y_test)      

print(classification_report(Y_test, pred_dtc))
print('Loading...')


rfc = RandomForestClassifier()    

rfc.fit(X_train, Y_train)
predict_rfc = rfc.predict(X_test)
rfc.score(X_test, Y_test)


print(classification_report(Y_test,predict_rfc))



# Optimizing by reducing the parameter n_estimators to 50. Optimizing by lower the depth of the tree. Using pararellized training with the n_jobs parameter.
gbc= GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1)   
# , n_jobs=-1
start= datetime.now()
gbc.fit(X_train, Y_train)
stop= datetime.now()
execution_time_gbc = start-stop
print('execution time gbc:', execution_time_gbc)
pred_gbc = gbc.predict(X_test)
gbc.score(X_test, Y_test)



#lightgbm test
d_train= lgb.Dataset(X_train, label=Y_train)
lgbm_params = {'learning_rate': 0.05,
               'boosting_type': 'gbdt',
               'objective': 'binary' ,
                'metric': ['auc', 'binary_logloss'],
                'num_leaves': 100,
                'max_depth': 10,} #Change num_leaves and test
start= datetime.now()
classifier = lgb.train(lgbm_params, d_train, 50) #50 iterations, learning rate and iterations are related.
stop= datetime.now()
execution_time_lgbm = start-stop
print('execution time lgbm:', execution_time_lgbm)



#Using profiles to see where the program takes a long time to execute.
cProfile.run('gbc.fit(X_train, Y_train)')
print(classification_report(Y_test, pred_gbc))

def output_label(n): 
    if n==0: 
        return "It is Fake News"
    elif n==1: 
        return "It is Real News"

def manual_testing(news):
    testing_news = {"text": [news]} #Corrected syntax for defining dictionary
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(stemming)


    new_x_test = new_def_test['text']
    new_xv_test = vectorizer.transform(new_x_test) #Assuming 'vectorizer' is your vectorizer object
    pred_lr = model.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    pred_lgbm= classifier.predict(new_xv_test)
    for i in range(0, new_x_test.shape[0]):
         if pred_lgbm[i] >= .5:
            pred_lgbm[i]=1
         else:
             pred_lgbm[i]=0
    return "\n\nLR prediction: {}  \nGBC Prediction: {}  \n RFC Prediction: {}\n LGBM Prediction: {}".format(output_label(pred_lr[0]), output_label(pred_gbc[0]), output_label(pred_rfc[0]),  output_label(pred_lgbm[0]))
print(r"Paste your news here:")
news_article=str(input())

print(manual_testing(news_article))