# Use a base Python image with a specific version
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTES=1

# Create the project directory
RUN mkdir -p /app

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy the project source code
COPY src /app/src

# Expose the application port
EXPOSE 7860

# Set the working directory
WORKDIR /app/src

# Start the application
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "7860", "--reload", "--reload-dir", "/app", "main:app"]