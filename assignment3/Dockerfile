FROM python:3.12-slim AS dependencies
WORKDIR /code
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

FROM dependencies
COPY helper_lib ./helper_lib
COPY main.py ./
COPY models/mnist_generator.pt ./models/mnist_generator.pt
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
