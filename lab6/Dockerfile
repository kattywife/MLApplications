FROM python:3.10-slim
WORKDIR /fast_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Мы запускаем из корня через модуль
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]