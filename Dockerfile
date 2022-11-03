FROM python:3.8.13

WORKDIR /app

ENV PORT 80

ADD app/application.py .
ADD app/requirements.txt .

RUN pip install -r requirements.txt

COPY . ./

CMD gunicorn -b 0.0.0.0:80 app.application:application
