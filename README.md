# 1stdaykit (a.k.a. FDK)
***Stupidly easy to use high-level Deep Learning toolkit for solving generic tasks.***

<p align="center">
  <img width="600" src="https://github.com/1stDayHack/1stdaykit/blob/master/misc/pic.png" />
</p>


## Getting Started 
Note that FDK comes in 2 flavours, FDK-Full and FDK-Lite. FDK-Lite is the same in everyway with FDK-Full, with the exception of <a href = "https://github.com/facebookresearch/detectron2"> Detectron2 </a> support. This means that FDK-Lite will not be capable of the following tasks:
* Object Detection
* Instance-Semantic Segmentation

If you do not need the above functionalities, then it would be advisable to just run with FDK-Lite for the sake of convenience. 


### Installation
FDK has been structured in such a way FDK-Full is simply FDK-Lite with a few extra modules. Hence, if you would like to install only FDK-Full, just follow through the entirety of the instructions below; for FDK-Lite, you can skip the steps on installing detectron2 below.


#### Easy Installation (Recommended; <a href = 'https://docs.anaconda.com/anaconda/install/'> Anaconda </a> Required)
```
#Installing FDK-Lite
git clone https://github.com/1stDayHack/1stdaykit.git
conda env create -f environment.yaml 
conda activate fdk

#Upgrading to FDK-Full (with detectron2)
#Check out instructions at https://github.com/facebookresearch/detectron2
```

#### Manual Installation
This option is for those that do not wish to install Anaconda on their machine. Please make sure that you have the correct version of python and pip installed beforehand, and do take note of the packages that will be installed upon execution of the following commands.

*Warning! Please be sure if you are doing this without any virtualenv like setup (e.g. Conda)!* 
```
#Installing FDK-Lite
git clone https://github.com/1stDayHack/1stdaykit.git
pip install -r requirements.txt

#Upgrading to FDK-Full (with detectron2)
#Check out instructions at https://github.com/facebookresearch/detectron2
```

## Development Logs and Notes

**To Do**:
* Add path-specification as init-argument.
* List authors and credits for original repos.
* Write docstrings.
* Clean up and shrink down base_libs dependencies.
* Complete requirement.txt
* Complete readme.txt
* Run tests, demonstrations and benchmarks.
* Include demo-video; e.g. super-res demo. 
* Add GPU / CPU option FLAGs for all model for option to toggle between manually.
* For Pytorch-Transformer models, give options to change model type (Light or Heavy) and list in docstring. (Simplicity first)
* For Pytorch-Transfomer Translation model, include options for option language pairs.
* Add light-weight 1stDayKit that has minimal dependencies requirement; e.g. no detectron2.
* Need separate installation instructions for detectron2; FDK-Full, no need for FDK-Lite.
* Installation support and testing on Ubuntu 18.04.
* Installation support and testing on Windows 10.


**Modules/Libs to Add**:
- [X] Detectron2
- [X] BlazeFace
- [X] DasiamRPN
- [X] Gen. Object Classifier
- [X] Gen. Object Classifier Lite
- [X] Super-Resolutor
- [X] MiDAS
- [X] PyTrans - Text Summarization 
- [X] PyTrans - Text Translation
- [X] PyTrans - Text Q&A
- [X] PyTrans - Text Generation Lite, Normal, Large, XL
- [X] PyTrans - Text Sentiment Analysis
- [X] Image Caption Generator
- [X] Deoldifier (Shelved)
