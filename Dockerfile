FROM python:3.7

RUN mkdir -p /app
WORKDIR /app

COPY /hotel_app .
COPY /models .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 5000

RUN export FLASK_APP=hotel_app

CMD ["flask", "run"]

