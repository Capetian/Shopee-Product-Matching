
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime	

RUN conda install -c conda-forge micromamba -y
RUN conda init bash
SHELL ["conda", "run", "/bin/bash", "-c"]


RUN micromamba install -c conda-forge notebook ipywidgets tqdm  faiss-cpu accelerate tensorboard -y
RUN micromamba install -c anaconda scikit-learn -y
RUN micromamba install -c fastai timm -y
RUN pip install fastai transformers madgrad kaggle 


EXPOSE 8888 6006
