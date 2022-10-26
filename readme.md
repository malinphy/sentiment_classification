# Sentiment Classification

Sentiment classification for twitter dataset.

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
