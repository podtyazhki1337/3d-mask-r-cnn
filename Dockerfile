FROM tensorflow/tensorflow:2.2.0-gpu

RUN adduser --disabled-password --gecos '' appuser

WORKDIR /workspace
ENV HOME=/workspace

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

COPY ./* .
RUN chown -R appuser:appuser /workspace

RUN pip install core/custom/tensorflow_nms_car_3d-0.1.0-cp36-cp36m-linux_x86_64.whl

CMD python -m main --help