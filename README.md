# Twitter-Senti-Meter
The proposed model is capable to detect hate or non-hate content automatically.The goal of this Twitter Senti-Meter is to classify tweets into two categories, hate speech or non-hate speech using machine learning Logistic Regression.Our project used a dataset CSV file from Kaggle containing 31,935 tweets. Out of 31,935 tweets 93% tweets were non-hate labeled and 7% tweets were hate-labeled. 
1. Importing the necessary libraries.
2. Reading the .csv file from Pandas and examining the head.
3. Cleaning the text data

Languages we speak and write are made up of several words often derived from one another and can contain words which don’t add meaning or context. In order to clean the data, we implemented 5 approaches.

-Stop Words Removal

-Greek Characters

-Slang Words

-Stemming

-Lemmatization

4. Splitting our data in the ratio of 93:7 for training and testing.
5. Transforming the words into feature vectors.
6. Creating the model and checking the score on training and test data.

Here we are using the LogisticRegression model because it’s easy to interpret in terms
of probability of the output. 

After training the model shows accuracy of 97%.


This project uses Flask as a framework for the frontend
