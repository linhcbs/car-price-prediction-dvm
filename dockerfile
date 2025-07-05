FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app/api.py /code/api.py

RUN mkdir -p /code/src

COPY ./src/data_preprocessor.py /code/src/data_preprocessor.py
COPY ./src/model.pkl /code/src/model.pkl
COPY ./src/preprocessor.pkl /code/src/preprocessor.pkl

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]