FROM python:3.9-slim

# インストール時のメッセージを非表示にするための環境変数
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ワーキングディレクトリの設定
WORKDIR /app

# requirements.txtとスクリプトをコピー
COPY requirements.txt /app/
COPY script.py /app/

# Pythonライブラリのインストール
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "script.py"]
