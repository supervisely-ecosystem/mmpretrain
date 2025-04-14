FROM supervisely/base-py-sdk:6.73.332

RUN pip install torch==2.1.0 torchvision==0.16.0 \
    --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install -U openmim && \
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html && \
    pip install mmengine

RUN pip install opencv-python-headless==4.5.5.64 && \
    pip install opencv-python==4.5.5.64

RUN pip install imgaug==0.4.0 && \
    pip install imagecorruptions==1.1.2

RUN pip install urllib3>=1.26.15 && \
    pip install requests-toolbelt==1.0.0 && \
    pip install yapf==0.40.1 && \
    pip install cachetools==4.2.4

RUN pip install importlib-metadata==8.5.0 && \
    pip install mat4py==0.6.0 && \
    pip install matplotlib==3.3.2 && \
    pip install model-index==0.1.11 && \
    pip install modelindex==0.0.2 && \
    pip install einops==0.8.1 && \
    pip install rich==13.4.2

RUN pip install supervisely==6.73.346