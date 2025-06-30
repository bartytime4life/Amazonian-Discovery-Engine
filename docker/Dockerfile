FROM python:3.11-slim-bullseye AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gdal-bin libgdal-dev proj-bin libproj-dev default-jdk && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java
WORKDIR /ade
COPY reproducibility/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
FROM python:3.11-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends gdal-bin libgdal-dev proj-bin libproj-dev default-jre && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
ENTRYPOINT ["/ade/reproducibility/run_pipeline.sh"]
