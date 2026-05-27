FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY trading_system/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('vader_lexicon')" || true
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; \
    BertTokenizer.from_pretrained('ProsusAI/finbert'); \
    BertForSequenceClassification.from_pretrained('ProsusAI/finbert')" || true

COPY trading_system /app/trading_system

RUN mkdir -p /app/trading_system/data/cache \
              /app/trading_system/models \
              /app/trading_system/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "-u", "-m", "trading_system.main"]
