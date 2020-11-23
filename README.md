# 1stdaykit (a.k.a. FDK)
***Stupidly easy to use high-level Deep Learning toolkit for solving generic tasks.***

<p align="center">
  <img width="800" src="misc/readme/nyc_final_1.gif" />
</p>


## Getting Started 
Note that FDK comes in 2 flavours, FDK-Full and FDK-Lite. FDK-Lite is the same in everyway with FDK-Full, with the exception of <a href = "https://github.com/facebookresearch/detectron2"> Detectron2 </a> support. This means that FDK-Lite will not be capable of the following tasks:
* Object Detection
* Instance-Semantic Segmentation

If you do not need the above functionalities, then it would be advisable to just run with FDK-Lite for the sake of convenience. 

<br>

### Sample Code

Object Detection and Segmentation with *detectron2*
```
#Import libs
from src.core.detect import Detector
from src.core.utils import utils
from PIL import Image
import os

#Instantiate detector
det = Detector(name="MyDet")

#Read Image and Prep
img = Image.open("misc/nyc_superres/out0166.jpg.jpg")

#Convert to cv2
img_cv = utils.pil_to_cv2(img)

#Infer and visualize
output = det.predict(img_cv)
out_img = det.visualize(img_cv,output,figsize=(18,18))
```
![](/misc/readme/detectron1.jpg)

<br>

Super-Resolution with *ESRGAN*
```
#Import libs
from src.core.super_res import SuperReser
from src.core.utils import utils
from PIL import Image
import os

#Initialization
super_res = SuperReser(name="SuperResssss")

#Read Image and Prep
img = Image.open("src/core/base_libs/ESRGAN/ny/rsz_china-street-changsha-city.jpg")

#Convert to cv2
img_cv = utils.pil_to_cv2(img)

#Infer and visualize
output = super_res.predict(img_cv)
super_res.visualize(img_cv,output,figsize=(20,20))
```
![](/misc/readme/superres1jpg.jpg)

<br>

Machine Translation with *MarianMT/Helsinki-NLP*
```
#Import libs
from src.core.translate import Translator_M
from src.core.utils import utils
import os

#Initialization
Trans = Translator_M(task='Helsinki-NLP/opus-mt-en-ROMANCE')

#Setup texts
text_to_translate = ['>>fr<< this is a sentence in english that we want to translate to french',
                     '>>pt<< This should go to portuguese',
                     '>>es<< And this to Spanish']
#Convert to cv2
img_cv = utils.pil_to_cv2(img)

#Infer and visualize
output = Trans.predict(text_to_translate)
Trans.visualize(text_to_translate,output)
```
![](/misc/readme/translate1.png)

<br>

### Installation
FDK has been structured in such a way FDK-Full is simply FDK-Lite with a few extra modules. Hence, if you would like to install only FDK-Full, just follow through the entirety of the instructions below; for FDK-Lite, you can skip the steps on installing detectron2 below.


#### Easy Installation (Recommended; <a href = 'https://docs.anaconda.com/anaconda/install/'> Anaconda </a> Required)
```
#Installing FDK-Lite
git clone https://github.com/1stDayHack/FDK.git
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
git clone https://github.com/1stDayHack/FDK.git
pip install -r requirements.txt

#Upgrading to FDK-Full (with detectron2)
#Check out instructions at https://github.com/facebookresearch/detectron2
```
<br> 

## Acknowledgement
*To be completed* 1stDayKit is built upon the works of many different authors and open-source projects. 1stDayKit serves primarily as a unifying wrapper that curates and simplifies said projects, and therefore relies on these projects very much; the contributions of their respective authors cannot be understated. I will be providing a complete list of references and acknowledgements here in the near future. Cheers!

<br>

## Development Logs and Notes

**To Do**:
* Add path-specification as init-argument.
* List authors and credits for original repos.
* Write docstrings.
* Complete readme.txt
* Run tests, demonstrations and benchmarks.
* For Pytorch-Transformer models, give options to change model type (Light or Heavy) and list in docstring. (Simplicity first)
* For Pytorch-Transfomer Translation model, include options for option language pairs.
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
