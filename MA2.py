####Question B


#' Download two attached datasets (Fake.csv and True.csv), which contains real and fake news. In the given
#' dataset, there is a total of five columns. Now build a Bi-LSTM model to detect fake news using TensorFlow
#' and other available libraries. Some hints are as follows:
#' 
#' 1. You might pre-process both the dataset using the Keras (for inspiration look into the exercises).
#' 
#' 2. You should also use Natural Language Toolkit (NLTK) library to handle “stopwords” in the dataset.
#' 
#' 3. For a better model, remove stopwords and remove words with 2 or fewer characters, after that split
#' data into test and train set.
#' 
#' 4. You might also need to create a tokenizer to tokenize the words and create sequences of tokenized words.
#' 
#' 5. You can call Bi-Directional LSTM from Keras and where fitting set Sigmoid and ReLU as activation
#' function, adam as an optimizer and Binary cross-entropy loss function should be used.
#' 
#' 6. During training set the batch size to 64 and the number of epochs to 2.
#' 
#' 7. Finally, print the model accuracy. If the predicted value is greater than 0.5 then it is a real news

import pandas as pd

True_df = pd.read_csv ("C:/Users/danie/Desktop/True.csv")
Fake_df = pd.read_csv ("C:/Users/danie/Desktop/Fake.csv")

#LSTM needs to be 3 dimentional
# https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/
# https://stackoverflow.com/questions/35169491/how-to-implement-a-deep-bidirectional-lstm-with-keras
# Add some code from homework in NLTK
