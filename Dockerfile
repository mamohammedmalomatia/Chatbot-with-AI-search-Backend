# Use an official Python runtime as a parent image
FROM python:3.11-slim
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
 
# Copy the .env file into the container
COPY .env .
 
# Copy the rest of your application code
COPY . .
 
EXPOSE 8000
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]