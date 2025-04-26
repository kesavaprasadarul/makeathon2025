
FROM hdgigante/python-opencv:5.0.0-alpha-ubuntu AS base

ARG REGENSBURG_PATH="nodeodm_out_4"
ARG REGENSBURG_IMAGE_PATH="dev_data/images"
ARG MUNICH_PATH="nodeodm_test_1"
ARG MUNICH_IMAGE_PATH="test_data"

# COPY "${REGENSBURG_PATH}" datasets\nodeodm_out_4
# COPY "${REGENSBURG_IMAGE_PATH}" datasets\images\regensburg
# COPY "${MUNICH_PATH}" datasets\nodeodm_test_1
# COPY "${MUNICH_IMAGE_PATH}" datasets\images\munich

FROM base AS builder

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN python3 -m pip config set global.break-system-packages true
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/app
EXPOSE 80
CMD ["fastapi", "run", "app/api.py", "--port", "80"]