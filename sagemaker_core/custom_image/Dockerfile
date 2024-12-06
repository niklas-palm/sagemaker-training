FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310

# Install system dependencies
RUN apt-get update && apt-get install -y 

# Install any other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy the script into the container
COPY train_extended.py .
 
# Setup the entry point
ENTRYPOINT ["python", "train_extended.py"]
