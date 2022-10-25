FROM python:3.9

ADD prediction_fast.py .

COPY sentiment_model.h5 .

COPY model.py .

COPY tv_layer.pkl .

COPY LE.pkl .

COPY requirements.txt .

# RUN pip install numpy pandas scikit_learn tensorflow

RUN pip install -r requirements.txt

CMD [ "python" , "prediction_fast.py" ]