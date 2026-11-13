FROM python:3.12-slim
WORKDIR /code/assignment4
COPY assignment4/requirements.txt ./
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY assignment4/helper_lib ./helper_lib
COPY assignment4/main.py ./
COPY assignment4/models ./models
COPY assignment3/models/mnist_generator.pt /code/assignment3/models/mnist_generator.pt
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
