FROM python:3.11-slim
WORKDIR /app
COPY pen_ai_bridge.py /app/pen_ai_bridge.py
COPY penai.apk /app/penai.apk
COPY penai.version /app/penai.version
ENV PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["python", "-u", "pen_ai_bridge.py"]
