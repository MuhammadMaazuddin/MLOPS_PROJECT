FROM python:3.10-slim

WORKDIR /app

COPY app.py .
COPY templates ./templates

RUN pip install --no-cache-dir flask

EXPOSE 5000

CMD ["python", "app.py"]
