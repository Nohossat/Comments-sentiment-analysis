FROM python:3.7

LABEL Author="Nohossat TRAORE"

ENV FLASK_APP "hotel_app"
ENV FLASK_ENV "development"
ENV FLASK_DEBUG True

RUN mkdir -p /app
WORKDIR /app/

COPY hotel_app hotel_app/
COPY models models/
COPY requirements_flask.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements_flask.txt


EXPOSE 5000

CMD flask run --host=0.0.0.0

