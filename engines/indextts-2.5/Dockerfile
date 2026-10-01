FROM python:3.12-slim
WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir --no-deps /workspace
RUN python -m pip install --no-cache-dir -r /workspace/engines/indextts-2.5/requirements.lock
ENV UNITTS_ENGINE_ID=indextts-2.5
CMD ["python", "-m", "unitts.services.engine_worker"]
