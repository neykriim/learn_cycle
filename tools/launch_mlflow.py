from mlflow.server import app
from flask import request, Response
import bcrypt
from functools import wraps
from getpass import getpass

# Конфигурация аутентификации
CREDENTIALS = {
    "admin": bcrypt.hashpw(getpass("Set password: ").encode('utf-8'), bcrypt.gensalt()).decode('utf-8')  # Замените на хеш вашего пароля (сгенерированный bcrypt)
}

def check_auth(username, password):
    """Проверка учетных данных"""
    if username in CREDENTIALS:
        hashed = CREDENTIALS[username].encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    return False

def authenticate():
    """Отправка запроса аутентификации"""
    return Response(
        'MLflow: Authentication required\n'
        'You must login with valid credentials', 401,
        {'WWW-Authenticate': 'Basic realm="MLflow Access"'})

def requires_auth(f):
    """Декоратор для защиты endpoint'ов"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Защищаем все endpoint'ы MLflow
for endpoint, view_func in app.view_functions.items():
    # Исключаем статические файлы и health-check
    if not endpoint.startswith('static') and endpoint != 'health_check':
        app.view_functions[endpoint] = requires_auth(view_func)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)