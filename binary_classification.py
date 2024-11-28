import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
# Import data
df, dataf, engdf, engdataf = pd.read_csv("sanna_nyheter.csv"), pd.read_csv("falska_nyheter.csv"), pd.read_csv("True.csv"), pd.read_csv("Fake.csv")


# Add label column
engdf['label'], engdataf['label']= 1 ,0
#Mergeing dataframes, concatinating(sammanfogar) objects within a list
frames, eng_frames= [df, dataf], [engdf, engdataf]
news_dataset, eng_news_dataset= pd.concat(frames), pd.concat(eng_frames)


#Remove unused columns from the datasets
news_dataset.drop([ 'ämne', 'titel'], axis=1, inplace=True)
eng_news_dataset.drop(['title', 'subject', 'date'], axis=1, inplace=True)

#shorten down the eng_news_dataset, selecting 10000 random rows from the eng_news_dataset
eng_news_dataset= eng_news_dataset.sample(n=10000) 


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


port_stem=PorterStemmer()
def stemming(content):
    # Remove urls, HTML tags, punctuation, digits, newline characters and stopwords
    stemmed_content = re.sub(r'https?://\S+|www\.\S+', '', content)
    stemmed_content = re.sub(r'<.*?>', '', content)
    stemmed_content = re.sub(r'[^\w\s]', '', content)
    stemmed_content = re.sub(r'\d', '', content)
    stemmed_content = re.sub(r'\n', '', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()   
    # stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('swedish') and stopwords.words('english') ]  
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
vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X= vectorizer.transform(X)
print(X)

# text will be used for X data, remaining Y datam stratify = Y , random state 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify= Y,random_state=2)
model= LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuraccy = accuracy_score(X_train_prediction, Y_train)    

print('Accuracy score of the training data: ', training_data_accuraccy)

X_test_prediction = model.predict(X_test)
test_data_accuraccy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data: ', test_data_accuraccy)


DTC = DecisionTreeClassifier()

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
print('Loading...')
gbc= GradientBoostingClassifier()   
gbc.fit(X_train, Y_train)
pred_gbc = gbc.predict(X_test)
gbc.score(X_test, Y_test)

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
    return "\n\nLR prediction: {}  \nGBC Prediction: {}  \n RFC Prediction: {}".format(output_label(pred_lr[0]), output_label(pred_gbc[0]), output_label(pred_rfc[0]))
print(r"Paste your fake news here(without formatting or paragraph division):")
news_article=str(input())

print(manual_testing(news_article))