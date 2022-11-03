FROM python:3.8.13

WORKDIR /app

ENV PORT 8081

ADD app/application.py .
ADD app/requirements.txt .

RUN pip install -r requirements.txt

COPY . ./

CMD gunicorn -b 0.0.0.0:8081 app.application:application
