FROM python:3.12.6-slim-bookworm as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PATH=/home/ftuser/.local/bin:$PATH
ENV FT_APP_ENV="docker"
ENV LD_LIBRARY_PATH /usr/local/lib

# Prepare environment
RUN mkdir /freqtrade \
    && apt-get update \
    && apt-get -y install --no-install-recommends sudo libatlas3-base curl sqlite3 libhdf5-serial-dev libgomp1 \
    build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 1000 -G sudo -U -m -s /bin/bash ftuser \
    && chown ftuser:ftuser /freqtrade \
    && echo "ftuser ALL=(ALL) NOPASSWD: /bin/chown" >> /etc/sudoers

WORKDIR /freqtrade

# Install TA-lib
RUN curl -L -o /tmp/ta-lib-0.4.0-src.tar.gz https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download \
    && tar xvzf /tmp/ta-lib-0.4.0-src.tar.gz -C /tmp \
    && cd /tmp/ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && rm -rf /tmp/ta-lib-0.4.0-src.tar.gz /tmp/ta-lib

# Install dependencies including Freqtrade
COPY --chown=ftuser:ftuser requirements.txt /freqtrade/
USER ftuser
RUN pip install --user --no-cache-dir --upgrade pip wheel \
    && pip install --user --no-cache-dir -r requirements.txt \
    && pip install --user --no-cache-dir freqtrade[hyperopt]

# Install and execute
COPY --chown=ftuser:ftuser . /freqtrade/

# Install Freqtrade UI
RUN freqtrade install-ui

# Make the startup script executable
COPY --chown=ftuser:ftuser start.sh /freqtrade/
RUN chmod +x /freqtrade/start.sh

# Expose port 8080 for the Freqtrade API
EXPOSE 8080

# Default command
CMD ["/bin/bash", "/freqtrade/start.sh"]