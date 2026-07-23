# Nava Vision AI Model Validation — conda-based image
FROM continuumio/miniconda3:24.7.1-0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PATH=/opt/conda/envs/model-validation/bin:$PATH

WORKDIR /app

# System libs used by OpenCV / video
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml ./
RUN conda env create -f environment.yml \
    && conda clean -afy

# App source
COPY app ./app
COPY run.py README.md requirements.txt ./
COPY sample_data ./sample_data
COPY scripts ./scripts

# Official YOLOX (needed for .pth loading)
RUN mkdir -p third_party \
    && git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git third_party/YOLOX

RUN mkdir -p uploads outputs/recordings

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/status')" || exit 1

CMD ["python", "run.py"]
