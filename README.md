**Email Spam Classification Using Naive Bayes**
---

**Overview**
---
This project focuses on email spam classification using the well-known Naive Bayes algorithm. Based on the Bag of Words model, the classifier is trained to distinguish between spam and non-spam emails. The dataset used for this project is sourced from the Kaggle Repository, and the performance is measured using metrics such as accuracy, precision, recall, and F1-score. The implementation is done using NLTK and Python. The results show an achieved accuracy of approximately 89.17%.

The project implements  NLP tasks  such as tokenization, lemmatization, stop word removal, the Multinomial Naive Bayes algorithm on the Bag of Words model, and model evaluation metrics.

**Introduction**
---
Emails are one of the primary communication mediums for companies and educational institutions in today’s digital world. Filtering spam from non-spam emails is challenging for humans, making automated email spam classification crucial. This project implements a Multinomial Naive Bayes classifier and evaluates its performance.


**Methodology**
---
***Dataset Preparation:***

Conducting a thorough examination of the dataset to understand the distribution of spam and non-spam emails.
Understanding the dataset and checking for null data if there exits then,
The dataset is split into training and testing sets using a 75-25 ratio to ensure a balanced evaluation of the model.
Stop words like "is," "am," "are," and "in" are removed as they do not contribute significantly to the classification task.
From the 3000 columns in the dataset, removing stop words reduced the features to 2867 columns.
For the final classifier, 228 columns with the highest frequency in the dataset were used to improve the model's performance.

***Training:***

The Multinomial Naive Bayes classifier is trained using the word frequency from the training set.
The model learns the probability distributions of words in spam and non-spam emails, enabling it to make predictions on unseen data.

**_Evaluation:_**

After training, the classifier is tested on the remaining 25% of the dataset, and its performance is evaluated using: metrics such as f1-score, accuracy, precision, and recall.

_This shows the result:_

               precision    recall  f1-score  

           0       0.95      0.90      0.92      
           1       0.76      0.88      0.82 
     accuracy                           0.89

**Dependencies**
---
*This project uses the following libraries and tools:*

**Python 3.12.2**: The core programming language used for implementing the classifier and handling data.

**NumPy**: Essential for numerical operations, especially in vectorizing text data.

**Pandas**: Used for loading and manipulating the dataset.

**Scikit-learn**: implementing the Multinomial Naive Bayes classifier, and evaluating the model through metrics like accuracy,f1-score, precision, and recall.

**NLTK**: For natural language processing tasks such as tokenization and stop word removal.

**Email Spam Collection Dataset**: The dataset contains 5172 emails, of which 1527 are spam and the remaining are non-spam.

