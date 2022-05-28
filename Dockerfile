FROM python:3.10
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && sed -i 's|security.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list && sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list && apt-get update && apt-get install -y ffmpeg
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
CMD ["python", "-u", "main.py", "config/config.json"]
