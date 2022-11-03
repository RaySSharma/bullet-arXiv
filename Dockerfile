FROM python:3.8.13

WORKDIR /app

ADD app/application.py .
ADD app/requirements.txt .

RUN pip install -r requirements.txt

COPY . ./

CMD gunicorn app.application:application
