FROM python:3.11-slim
WORKDIR /app
COPY pen_ai_bridge.py /app/pen_ai_bridge.py
COPY designer.html /app/designer.html
COPY config.json /app/config.json
COPY penai.apk /app/penai.apk
COPY penai.version /app/penai.version
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
EXPOSE 10000
CMD ["python", "-u", "pen_ai_bridge.py"]
