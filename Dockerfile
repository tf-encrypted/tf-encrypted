FROM python:3.6.3

RUN mkdir -p /usr/src/tf-encrypted \
    && pip install --upgrade \
          https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

WORKDIR /usr/src/tf-encrypted

COPY . .
RUN pip install -e .

ENTRYPOINT ["/bin/bash"]
