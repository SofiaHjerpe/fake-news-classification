import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Import data
dftrue, dffake,engdffake, engdftrue = pd.read_csv("sanna_nyheter.csv"), pd.read_csv("falska_nyheter.csv"), pd.read_csv("fake.csv"), pd.read_csv("true.csv")


# Add label column and values to english datasets
engdffake['label'] = 0
engdftrue['label']= 1


# Mergeing true and false dataframes
frames= [dftrue, dffake]
eng_frames=[engdftrue, engdffake]
news_dataset, eng_news_dataset= pd.concat(frames), pd.concat(eng_frames)
# Remove unused columns
news_dataset.drop([ 'ämne', 'titel'], axis=1, inplace=True)
eng_news_dataset.drop(['date','subject', 'title'], axis=1, inplace=True)
#Mergeing swedish and english dataframes
eng_swe_frames = [news_dataset, eng_news_dataset]
dataset = pd.concat(eng_swe_frames)


#Reshuffeling
dataset= dataset.sample(frac=1)
#Resetting index
dataset.reset_index(inplace=True)
dataset.drop(['index'], axis=1, inplace=True)
#Fill null values with empty strings
dataset = dataset.fillna('')
dataset['content']= dataset['text'] 

# Seperate data and label, X is the content and Y is the labels.
X=dataset.drop(columns='label', axis=1)
Y= dataset['label']


# Remove urls, HTML tags, punctuation, digits and newline characters
def shorten_down(content):
    shorten_content = re.sub(r'https?://\S+|www\.\S+', '', content)
    shorten_content = re.sub(r'<.*?>', '', content)
    shorten_content = re.sub(r'[^\w\s]', '', content)
    shorten_content = re.sub(r'\d', '', content)
    shorten_content = re.sub(r'\n', '', content)
    shorten_content  = shorten_content.lower()
    shorten_content  = shorten_content.split()   
    shorten_content = ' '.join(shorten_content )  
    return shorten_content 



# It will process a lot of words.
dataset['content'] = dataset['content'].apply(shorten_down)
print('Loading...')
# Store the content in X and label in Y variabel
X=dataset['content'].values
Y=dataset['label'].values


# Convert textdata to numerical data. 
vectorizer=TfidfVectorizer(max_features=1000)
vectorizer.fit(X)
X= vectorizer.transform(X)


# Splits arrays into random train and test subsets. Test size represents the proportion of the dataset to include in test split. Random_state effects the shuffeling of the data before the split is applied.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)


# LogisticRegression uses the sigmoidfunction. The sigmoidfunktion ensures that output are between 1 and 0. It uses linear combination of input and converts it into probabilities. 
# I choose this model because it specifically made for binary classification and because it is fast. It is also good at making biased predictions based on imbalenced datasets.
model= LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuraccy = accuracy_score(X_train_prediction, Y_train)    

print('Accuracy score of the training data: ', training_data_accuraccy)

X_test_prediction = model.predict(X_test)
test_data_accuraccy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data: ', test_data_accuraccy)
print('Logistic regression:',classification_report(Y_test, X_test_prediction))



def output_label(n): 
    if n==0: 
        return "It is Fake News"
    elif n==1: 
        return "It is Real News"
    


def manual_testing(news):
    testing_news = {"text": [news]} 
    test_df = pd.DataFrame(testing_news)
    test_df['text'] = test_df['text'].apply(shorten_down)


    new_x_test = test_df['text']
    new_xv_test = vectorizer.transform(new_x_test)
    pred_lr = model.predict(new_xv_test)
    return "\n\nPrediction: {} ".format(output_label(pred_lr[0]))



while True: 
   print(r"Paste your news here (the text in your article):")
   news_article=str(input())
   if(news_article != ''):
      print(manual_testing(news_article))
   else:
       break;