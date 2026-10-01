FROM python:3.12-slim
WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir --no-deps /workspace
RUN python -m pip install --no-cache-dir -r /workspace/engines/vibevoice/requirements.lock
ENV UNITTS_ENGINE_ID=vibevoice
CMD ["python", "-m", "unitts.services.engine_worker"]
