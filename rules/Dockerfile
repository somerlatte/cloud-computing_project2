FROM python:3.10-slim

WORKDIR /app


COPY ./rules /app


RUN pip install --no-cache-dir pandas fpgrowth-py


RUN mkdir -p /mnt/shared


EXPOSE 52041


CMD ["python", "app.py"]
