#
FROM ubuntu:22.04

#
RUN apt-get update && apt-get install -y \
    libgl1 libgl1-mesa-glx libglib2.0-0 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 
WORKDIR /workingdirectory

#
COPY requirements.txt /workingdirectory/
RUN pip install -r requirements.txt \
    && rm requirements.txt
    
