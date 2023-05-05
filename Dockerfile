# base image
FROM tensorflow/tensorflow:2.11.0-gpu

# user and group ID arguments
ARG USER_ID=1000
ARG GROUP_ID=1000

# add group and user
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# copy requirements
WORKDIR /tmp
COPY requirements.txt /tmp

# install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
