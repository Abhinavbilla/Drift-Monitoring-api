FROM python:3.10-slim

WORKDIR /app

# Install system dependencies: nginx, supervisor, and build tools
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    gcc \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Apply the patch for streamlit-google-auth (if needed)
RUN cp /app/patched_init.py /usr/local/lib/python3.10/site-packages/streamlit_google_auth/__init__.py 2>/dev/null || true

# Configure nginx to proxy requests to the right backend
RUN echo 'server { \
    listen 80; \
    location /api/ { \
        proxy_pass http://localhost:8000/; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
    location / { \
        proxy_pass http://localhost:8501/; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
    } \
}' > /etc/nginx/sites-enabled/default

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port 80 (nginx)
EXPOSE 80

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]