FROM python:3.5

RUN mkdir -p /usr/src/tf-encrypted \
    && pip install --upgrade pip \
    && pip install --upgrade tensorflow==1.11.0

WORKDIR /usr/src/tf-encrypted

COPY . .
RUN make bootstrap

EXPOSE 4440

ENTRYPOINT ["python", "-u", "bin/serve"]
