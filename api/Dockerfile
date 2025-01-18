FROM python:3.9-slim

WORKDIR /app

COPY ./api /app

RUN pip install --no-cache-dir Flask

EXPOSE 52041

CMD ["python", "app.py"]
