
FROM hdgigante/python-opencv:5.0.0-alpha-ubuntu AS base

ARG REGENSBURG_PATH="nodeodm_out_4"
ARG REGENSBURG_IMAGE_PATH="dev_data/images"
ARG MUNICH_PATH="nodeodm_test_1"
ARG MUNICH_IMAGE_PATH="test_data"

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN python3 -m pip config set global.break-system-packages true
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/app
EXPOSE 80
CMD ["fastapi", "run", "app/api.py", "--port", "80"]