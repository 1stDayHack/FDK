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
1stDayKit (FDK) is built upon the works of many different authors and open-source projects. 1stDayKit serves primarily as a unifying wrapper that curates and simplifies said projects, and therefore relies on these projects very much; the contributions of their respective authors cannot be understated. They are as follow:

**Detectron2**
```
@misc{wu2019detectron2,
author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                Wan-Yen Lo and Ross Girshick},
title =        {Detectron2},
howpublished = {\url{https://github.com/facebookresearch/detectron2}},
year =         {2019}
}
```

**BlazeFace**
```
@misc{bazarevsky2019blazeface,
      title={BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs}, 
      author={Valentin Bazarevsky and Yury Kartynnik and Andrey Vakunov and Karthik Raveendran and Matthias Grundmann},
      year={2019},
      eprint={1907.05047},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**Torchvision Classifer Models - Wide ResNet & MobileNetv2**
```
@incollection{NEURIPS2019_9015,
title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {8024--8035},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}

@misc{zagoruyko2017wide,
      title={Wide Residual Networks}, 
      author={Sergey Zagoruyko and Nikos Komodakis},
      year={2017},
      eprint={1605.07146},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{sandler2019mobilenetv2,
      title={MobileNetV2: Inverted Residuals and Linear Bottlenecks}, 
      author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
      year={2019},
      eprint={1801.04381},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**DaSiamRPN**
```
@InProceedings{Yan_2020_CVPR,
author = {Yan, Bin and Wang, Dong and Lu, Huchuan and Yang, Xiaoyun},
title = {Cooling-Shrinking Attack: Blinding the Tracker With Imperceptible Noises},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

**MiDaS**
```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

**ESRGAN**
```
@misc{wang2018esrgan,
      title={ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks}, 
      author={Xintao Wang and Ke Yu and Shixiang Wu and Jinjin Gu and Yihao Liu and Chao Dong and Chen Change Loy and Yu Qiao and Xiaoou Tang},
      year={2018},
      eprint={1809.00219},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**HugginFace Transformers**
```
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```

**Show-Attend-Tell Caption Generator**
```
@misc{xu2016show,
      title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention}, 
      author={Kelvin Xu and Jimmy Ba and Ryan Kiros and Kyunghyun Cho and Aaron Courville and Ruslan Salakhutdinov and Richard Zemel and Yoshua Bengio},
      year={2016},
      eprint={1502.03044},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

*Note: If I missed out any relevant parties, please do send me a quick ping at my email address or help by amending through a PR. Much thanks!
<br>

## Development Logs and Notes

**To Do**:
- [ ] Complete readme.txt
- [ ] Write docstrings.
- [ ] For Pytorch-Transfomer Translation model, include options for option language pairs.
- [x] List authors and credits for original repos.
- [x] Run tests, demonstrations and benchmarks.
- [x] For Pytorch-Transformer models, give options to change model type (Light or Heavy) and list in docstring. (Simplicity first)
- [x] Need separate installation instructions for detectron2; FDK-Full, no need for FDK-Lite.
- [x] Installation support and testing on Ubuntu 18.04.
