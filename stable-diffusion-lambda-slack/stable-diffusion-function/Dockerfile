FROM public.ecr.aws/lambda/python:3.9

RUN python3.9 -m pip install --no-cache-dir \
        diffusers[torch] \
        transformers \
        scipy \
        accelerate \
        torch --extra-index-url https://download.pytorch.org/whl/cpu \
        slack_bolt

RUN yum install -y \
        mesa-libGL-devel \
        mesa-libGLU-devel \
        libpng-devel && \
    yum clean all

COPY ./model/ /model/

COPY app.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
