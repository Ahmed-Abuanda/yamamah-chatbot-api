# FROM hub.leandevclan.com/python:3.10
FROM python:3.11

RUN pip install --upgrade pip
 
COPY ./requirements.txt .
RUN pip install -r requirements.txt
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the current directory contents into the container at /app
COPY . /app
 
 
# Expose the port that FastAPI will run on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
