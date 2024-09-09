# Use the official Python 3.8 slim image as the base image
FROM python:3.11-slim

# Set the working directory within the container
WORKDIR /churn_project

RUN apt-get update && apt-get install -y --no-install-recommends gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the necessary files and directories into the container
COPY static/ templates/ FinalDTModel_07Sept2024.pkl app.py requirements.txt /churn_project/
COPY templates/ /churn_project/templates/
COPY static/ /churn_project/static/
COPY FinalDTModel_07Sept2024.pkl app.py requirements.txt /churn_project/

COPY requirements.txt .
# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask application
EXPOSE 5000

# Define the command to run the Flask application using Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "-w", "4"]