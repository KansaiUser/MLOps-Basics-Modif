# Use an official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# aws credentials configuration
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Copy the requirements file into the container
COPY requirements_inference.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_inference.txt

# Copy the app code into the container
COPY . /app/
RUN rm -rf /app/models  # Ensure the models directory is excluded

# Initialize DVC repository
RUN dvc init --no-scm

# configuring remote server in dvc
RUN dvc remote add -d model-store s3://models-dvc-modified-version/trained_models/

RUN cat .dvc/config
# pulling the trained model

RUN dvc pull models.dvc


# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD bash -c "dvc pull models.dvc && uvicorn app:app --host 0.0.0.0 --port 8000"
# CMD ["bash", "-c", "dvc pull models.dvc && uvicorn app:app --host 0.0.0.0 --port 8000"]
