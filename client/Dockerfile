FROM python:3.10-slim

WORKDIR /app

COPY client.py ./
COPY data ./data

RUN pip install flwr scikit-learn tensorflow pandas numpy

CMD ["python", "client.py"]
