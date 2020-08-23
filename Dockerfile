FROM python:3.7

RUN mkdir -p /app
WORKDIR /app

COPY /hotel_app ./hotel_app
COPY /models ./models

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "./hotel_app/app.py"]

