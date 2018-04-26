FROM python:3.6.5-jessie

# Add and install Python modules
ADD requirements.txt /src/requirements.txt
RUN cd /src; pip install -r requirements.txt

# Bundle app source
ADD . /src

# Expose
EXPOSE 5000

# Run
CMD ["python", "/src/application.py"]
