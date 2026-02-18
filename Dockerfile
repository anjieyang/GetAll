FROM python:3.12-slim-bookworm

# Accept proxy args for build time
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY}

# Install system deps + Node.js 20 for the WhatsApp bridge
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg git && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y gnupg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p getall bridge && touch getall/__init__.py && \
    uv pip install --system --no-cache . && \
    rm -rf getall bridge

# Copy the full source and install
COPY getall/ getall/
COPY bridge/ bridge/
COPY workspace/ workspace/
RUN uv pip install --system --no-cache .

# Build the WhatsApp bridge
WORKDIR /app/bridge
RUN npm install && npm run build
WORKDIR /app

# Install Playwright + Chromium for headless chart rendering
RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps chromium && \
    rm -rf /tmp/*

# Clear proxy for runtime
ENV http_proxy="" https_proxy=""

# Create config directory
RUN mkdir -p /root/.getall

EXPOSE 18790

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD getall status || exit 1

ENTRYPOINT ["getall"]
CMD ["gateway"]
