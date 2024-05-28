FROM tensorflow/tensorflow:2.2.0-gpu

RUN adduser --disabled-password --gecos '' appuser

WORKDIR /workspace
ENV HOME=/workspace
ENV TF_CPP_MIN_LOG_LEVEL='3'

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

COPY core/custom_op/tensorflow_nms_car_3d-0.1.0-cp36-cp36m-linux_x86_64.whl .
COPY requirements.txt .
RUN chown -R appuser:appuser /workspace

RUN pip install tensorflow_nms_car_3d-0.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install -r requirements.txt

CMD python -m main --help