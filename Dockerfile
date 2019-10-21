FROM python:3.5-stretch

RUN mkdir -p /usr/src/tf-encrypted \
    && pip install --upgrade pip

WORKDIR /usr/src/tf-encrypted

COPY . .

RUN make bootstrap

EXPOSE 4440

ENTRYPOINT ["python", "-u", "bin/serve"]
