# Imagen base liviana de Python
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar archivo de dependencias (puedes ponerlo en la raíz o en api/)
COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar SOLO la carpeta api (no el training ni data)
COPY api/ ./api

WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]