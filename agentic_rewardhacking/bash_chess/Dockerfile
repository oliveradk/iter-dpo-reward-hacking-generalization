FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
        stockfish \
        procps \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'PATH="$PATH:/usr/games"' > /etc/profile.d/usr-games.sh

RUN pip install --no-cache-dir chess==1.11.2

WORKDIR /env
COPY game.py /env/game.py
RUN chmod +x /env/game.py && mkdir -p /env/game

ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/usr/games

CMD ["tail", "-f", "/dev/null"]
