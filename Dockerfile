# Image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail dans l'image
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le reste du code dans le conteneur
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.enableCORS=false", "--server.port=8501"]
