FROM python:3.9-slim-bullseye

WORKDIR /code/rest_api

COPY ./requirements.txt /code/requirements.txt

COPY ./models /code/models

COPY ./rest_api /code/rest_api

COPY ./yolov5 /code/yolov5

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]