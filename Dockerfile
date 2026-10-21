# Stage 1: build the React frontend (served by nginx at /, talking to the API at /api).
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_GOOGLE_CLIENT_ID
ENV VITE_API_BASE_URL=/api VITE_GOOGLE_CLIENT_ID=$VITE_GOOGLE_CLIENT_ID
RUN npm run build

FROM python:3.12-slim

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
# torch/torchvision pulled from the CPU-only wheel index to avoid bundling
# CUDA libraries we'll never use in this container — keeps the image smaller.
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Apply the patch for streamlit-google-auth (if needed)
RUN cp /app/patched_init.py /usr/local/lib/python3.12/site-packages/streamlit_google_auth/__init__.py 2>/dev/null || true

COPY --from=frontend /frontend/dist /app/frontend/dist

# nginx: React app at /, API at /api/, the old Streamlit dashboard at /streamlit/.
# client_max_body_size covers a 200MB table plus a 500MB image ZIP; table jobs
# run in the background, so request timeouts only need to cover the upload.
COPY nginx.conf /etc/nginx/sites-enabled/default

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port 80 (nginx)
EXPOSE 80

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]