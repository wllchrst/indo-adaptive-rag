FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy the rest of the source code
COPY . .

CMD ["python", "-m", "main"]