# Sentiment Classification

Overview :
This study model is constructed as feed forward neural network
using  Tensorflow/Keras framework on Twitter dataset. Model composed of 3 dense layers 
with relu activation function. As final layer a dense layer with 
softmax activation function. 
Performance of the model can be seen in evaluation section.
Prediction file, model ,dictionaries ,model weights were dockerized. Docker file can be pulled from docker hub.

Data
----
Twitter dataset on kaggle : https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

File Description :
----
- LE.pkl : pickle file for Label encoder 
- tv_layer.pkl : pickle file for Tensorflow text vectorization
- model.py : nueral network model
- prediction_fast.py : prediction file for deployment purpose 
- sentiment_analysis.py : training file 
- sentiment_model.h5 : model weights 
- requirements.txt : dependencies 

Docker
----
Docker pull command:
```
docker pull maliphy/sentiment_class
```
Docker run for prediction:
```
docker run -t -i maliphy/sentiment_class:v1
```
Evaluation
----
Twitter sentiment analysis
```
                   precision recall  f1-score   support

  Irrelevant       0.97      0.98      0.97       172
    Negative       0.97      0.99      0.98       266
     Neutral       0.99      0.96      0.98       285
    Positive       0.98      0.97      0.97       277

    accuracy                           0.97      1000
   macro avg       0.97      0.98      0.97      1000
weighted avg       0.98      0.97      0.97      1000
```
