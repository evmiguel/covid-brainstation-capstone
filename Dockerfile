FROM continuumio/anaconda3:2024.06-1

WORKDIR /app
COPY data data
COPY models models
COPY src src
COPY lib lib
COPY conda.yml .

RUN conda env create -f conda.yml
RUN echo "conda activate capstone" >> ~/.bashrc
SHELL ["conda", "run", "-n", "capstone", "/bin/bash", "-c"]

EXPOSE 8501

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "capstone", "streamlit", "run", "src/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

