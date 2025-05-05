# Use the official TensorFlow Docker image with version 2.18.0
FROM tensorflow/tensorflow:2.18.0

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Copy the packages directory into the container
COPY packages/ /packages/

# Force reinstall blinker to avoid uninstall issues
RUN pip install --no-cache-dir --ignore-installed blinker

# Install dependencies from the local packages directory
RUN pip install --no-cache-dir --find-links=/packages --timeout=120 -r requirements.txt

# Copy the application and model files
COPY app/ app/
COPY models/ /app/models/

# Expose the port that the Flask app runs on
EXPOSE 5000
ENV TF_ENABLE_ONEDNN_OPTS=0

# Command to start the Flask app
CMD ["python", "app/app.py"]