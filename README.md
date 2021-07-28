# SOC 2021
## NLPlay with Transformers
### Mentors - Tezan Sahu & Shreya Laddha

I am Shreedhar Malpani (200020132) and this is my repository for my SOC 2021 project - 'NLPlay with Tranformers'

 ## Phase 1- (22th March to 15th April)
\
The first week started with getting familiar with pytorch by implementing basic tensor operations and a feed forward neural network for classification task on MNIST-dataset. Along, with that we got introduced to a very powerful text processing library in python which is nltk , it provides utility functions for data preprocessing/cleaning task such as removing stopwords,punctuations, Stemming and Lemmatization task , tokenisation of corpus, etc.
\
<br/>
Pytorch being more pythonic, involves Object Oriented Prograaming Paradigm and hence the implementation of the neural Architecture was more intuitive and expressive. Since a computer program cannot take string/word as input , hence the corpus of text used in training must be converted into numeric form which is done using word vector or word embedding. 
\
<br/>
## Implementation:
\
We were supposed to code a feed forward neural net model for Sentiment analysis( Classification Task) on [IMDB moview review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from kaggle. I started by first importing and conducting Exploratory data Analysis(EDA) and got the gist of the data distribution. Then used nltk and re library for preprocessing task, Started with a 2 hidden layer neural net with 500 neurons each and used the cross entropy loss function and Adaptive moment estimation(adam) optmiser for back propagation of the loss function updating the parameters of the network. The f1 score was not too pleasing, obviously it required _hyperparameter tuning_, I finally resorted to 3 hidden layer model with a softmax activation function in the output layer. 
\

## Phase 2-  (17th May to 31st May)

Keeping the preprocessing process same, made changes in the nn Module sub-class's computation graph to incorporate GRU and LSTM architecture which solve the bottleneck of RNNs, i.e. Long term dependency Integration through Gated Mechanism which restricts unncessary information and passes relevant information. LSTM and GRU were able to solve the vanishing/exploding gradient problem of RNN and also had memory but took a lot of time to train

## Implementation:
\
We were supposed to now implement our sentiment analysis model using RNN (Recurrent Neural Networks), LSTM ( Long Short Term Memory) and GRU (Gated Recurrent Unit) and achieved accuracies of 82.78 %, 86.20 %, 85.45 % respectively.
\

## Phase 3- (4th July to 17th July)

Got introduced to State-of-the Art Encoder-Decoder based Transformer models.It solved some major bottlenecks in using Recurrent neural architecture like LSTM,GRU. LSTM/GRU, i.e the Gradient vanishing and EXploding and one shot parallel data feeding. The mathematics behind self attention mechanism and it's importance is really useful not only for NLP tasks but also vision and other downstream tasks. Going Ahead, I learnt about different architectures like BERT which is a only Encoder transformer. There have been lots of other architecture created out of BERT naming a few-  BERT & DistilBERT. DistilBERT is a distiled version of BERT with almost half the parameters but fast and gives similar types of results. 

### Implementation:- 
Created a Sentiment Classifier with BERT architecture( bert-base-uncased)

The  HuggingFace library proviides various open-source pre-trained models for quick integration with easy to use API and highly user-friendly. Transformers package Documentation was really helpful while using the Transformer models for transfer learning task(Fine tuning)

## Concluding Remarks:

Doing the SOC was a great experience and I actually got to learn a lot under my mentors and with my co-mentees.
\
Thank you!
