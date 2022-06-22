FROM condaforge/mambaforge:4.10.3-7

USER root
ENV HOME /root

RUN pip install -U albumentations && \
    pip cache purge && \
    mamba update -y mamba && \
    mamba install -y pytorch torchvision torchaudio cudatoolkit=10.2 pytorch-metric-learning tensorboard -c metric-learning -c pytorch && \
    conda clean -tipsy && \
    conda clean -afy

COPY . ${HOME}/checkbox_classification
WORKDIR ${HOME}/checkbox_classification
CMD ["bash"]
