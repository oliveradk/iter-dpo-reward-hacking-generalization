# Dockerfile for the codecontests training-env sandbox.
#
# This environment is intentionally vulnerable to reward hacking for
# research purposes: pytest is installed so the conftest.py hack is
# exploitable.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# pytest is required (the harness runs `pytest test_solution.py`); the
# conftest hack also relies on pytest's report internals.
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-timeout>=2.0.0

RUN mkdir -p /workspace
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

CMD ["tail", "-f", "/dev/null"]
