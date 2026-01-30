FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies required for Postgres and Pandas
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire project structure
COPY . .

# Create the sandbox directory and ensure it is writable
# This is where the CSVs and Plots will live
RUN mkdir -p /code/app/sandbox && chmod 777 /code/app/sandbox

# Add /code to PYTHONPATH so 'import app.state' works correctly
ENV PYTHONPATH=/code
# Prevent Python from writing .pyc files to the container
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure logs are sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]