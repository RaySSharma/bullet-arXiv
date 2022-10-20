FROM python:3.8

WORKDIR /app

ENV PORT 80

ADD app/application.py .
ADD app/requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./application.py"]