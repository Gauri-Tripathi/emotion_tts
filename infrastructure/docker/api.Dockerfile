FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir ".[service]"
EXPOSE 8000
CMD ["unitts-api"]
