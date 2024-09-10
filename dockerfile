# Use the official Python 3.8 slim image as the base image
FROM python:3.11-slim

# Set the working directory within the container
WORKDIR /churn_project

RUN apt-get update && apt-get install -y --no-install-recommends gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the necessary files and directories into the container
COPY static/ templates/ uploads/ FinalDTModel_07Sept2024.pkl app.py requirements.txt /churn_project/
COPY templates/ /churn_project/templates/
COPY uploads/ /churn_project/uploads/
COPY static/ /churn_project/static/
COPY FinalDTModel_07Sept2024.pkl app.py requirements.txt /churn_project/

COPY requirements.txt .
# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN echo '#!/bin/bash\n\
gunicorn app:app -b 0.0.0.0:8060 -w 4 --log-level debug --error-logfile - --access-logfile - --capture-output' > /churn_project/start.sh && \
chmod +x /churn_project/start.sh

# Expose port 8080 for the Flask application
EXPOSE 8060

# Run the start script
CMD ["/churn_project/start.sh"]