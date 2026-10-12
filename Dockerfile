# 1. Start from an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working folder inside the container
WORKDIR /app

# 3. Copy only requirements first, then install them.
# Docker caches this layer, so libraries are not reinstalled
# every time you change your code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project (code, models, processed data)
COPY . .

# 5. Document the port Streamlit listens on
EXPOSE 8501

# 6. Command that runs when the container starts.
# 0.0.0.0 makes the app reachable from outside the container.
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
