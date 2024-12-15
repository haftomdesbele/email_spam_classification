import nltk,re
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from nltk import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
def token_usef(message):
    tokenizer=nltk.RegexpTokenizer(r"\w+")
    token=tokenizer.tokenize(message)
    token_lower=[t.lower() for t in token]
    lemma=WordNetLemmatizer()
    token_lemma=[lemma.lemmatize(t) for t in token_lower]
    print(token_lemma)
    token_usefl=[]
    for t in token_lemma:
        if t not in  st:
            token_usefl.append(t)
    return token_usefl
st=stopwords.words('english')
data=pd.read_csv('emails.csv',encoding='latin-1')
print(data)
print("Null values in the data sets",data.isnull().sum()) #pre procesing thedata sets  
cols=data.columns
cols_st=[]
for t in cols:
    if t in st:
        cols_st.append(t)

print("The shape of the data set is ",data.shape)
data=data.drop('Email No.',axis=1)
data=data.drop(cols_st,axis=1)
print("The shape of data after dropping stop words",data.shape)
columns_sum=data.sum()
threshold=1000
top_tr_cols=columns_sum[columns_sum>=threshold].index
data_top=data[top_tr_cols]
print('=----------------------------------------------=')
print(data_top)
print(data_top.shape)


y=data_top['Prediction']
X=data_top.drop('Prediction',axis=1)
print(X.head(5))
print(y.shape)
sn.countplot(data=data,x='Prediction')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

nb_class=MultinomialNB()
nb_class.fit(X_train,y_train)
y_pred=nb_class.predict(X_test)

#evaluate the performance 
accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)
print('The accuacy is=',accuracy*100 )
print(report)
