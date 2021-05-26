# ScriptieOffensiveLanguage
This will be my repository for my scriptie. This will make for good version control.

## Preface

## LinearSVC
We wanted to use a simple yet thorough machine learning model for our baseline. We looked at Support Vector Machines and finally decided to use a Linear Support Vector Classification (LinearSVC) model to compare the other two models to. We believed this to be a good and simple machine learning model. State-of-the-art results were already achieved by SVM's for many text classification tasks \citep{zampieri2018language}. In addition to the preprocessing steps we took for all our classifiers we also used the following features:
    - bag of words
    - bigrams
    - remove punctuation
    - tokenization
Firstly we put the text in bag of words to make data processing and feature extraction easier. However, we did not only put the unigrams in the bags but also the bigrams of the sentence. By using a combination of unigrams and bigrams we ensured that as little information of the sentence was lost. We also decided to remove the punctuation from the sentence, since this did not contribute to the explicitness features.

## BiLSTM
For another machine learning model we decided to experiment with a Bidirectional Long Short Term Memory (BiLSTM) model.
Our model starts with the input layer where the input is raw text, this is then transformed to an embedding layer as is required by neural networks. In the embedding layer all the words are converted to real value vectors using the Coosto word embeddings model. In this layer, we also pad the sentences so they are all of equal length. The third layer is the bidirectional LSTM layer where the vectors are processed both backwards and forward. The dropout layer is a regularization layer to counter the overfitting of data. Lastly we have a dense layer where the output of the previous layer is transformed in the label output we need. 

## CNN
We also used A Convolutional Neural Network (CNN) to test our hypothesis.
Our CNN model will very similarly to our BiLSTM model start with an input layer of raw text. However, this raw text will be transformed to an embedding matrix where the words are converted with the same Coosto word embeddings model. Where the BiLSTM model takes a sequential input for the embedding layer, the CNN model takes a matrix as input. Next up, it will go through the Conv1D layer, here the matrix from the previous layer will be weighed and transformed into a smaller matrix. Using the MAxPooling1D layer we make this matrix even smaller by grabbing the maximum of a 2 x 2 matrix and sending this to the output layer. By using this the initial matrix will be smaller and more dense of information. The following layer called Flatten will transform this matrix into a 1-dimensional array for the next layer. This 1-dimensional array will be converted back into the output labels in the Dense layer.
