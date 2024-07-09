
# AI FOR GOOD: Patient clustering with VAE-SOM AND DPSOM 
### Felix Steckenbiller & Paul Gramlich - Ludwig-Maximilians-Universität München

This project contains the application of a new approach (self-organizing maps) to patient clustering. 

### Dataset 

#### `/Data/preprocessing.py`
We used a Lower Back Pain (LBP) patient dataset, that was preprocessed by us to convert categorical variables into numerical values. 
Furthermore, NAs were removed and replaced with `0` and scaled the data from `[0,1]`.
Scaled and unscaled versions for plotting can be found in `/Data/LBP`.

#### `/Data/processing.py`
Then the dataset was processed to fit the requirements of the models. 
The provided script is just an excerpt of the code that was then injected into the training files of the models.

### Models
We used 2 models and 1 baseline model: 
* VAE-SOM (original model and paper: https://github.com/ratschlab/SOM-VAE) `/SOMVAE`
* DPSOM (original model and paper: https://github.com/ratschlab/dpsom) `/DPSOM`
* Baseline Model: K-MEANS++ `/KMEANS`

### Technical requirements
#### **DPSOM and KMEANS**: 
Current Python version (3.12.X) and current installation of all packages that were used

#### **SOM-VAE**: Python 3.6 with CUDA 9 and CuDNN 7.0.5 
*Installation of right CuDNN-version:*
* If needed: find installed CuDNN-version with `dpkg -l | grep cudnn` and deinstall with `dpkg --revome your-cudnn-version`
* Change directory: `cd /SOMVAE`
* Install CuDNN with `dkpg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb`
* Fix dependencies with `apt-get install -f`
  
*Packages:*
* Change directory: `cd /SOMVAE`
* Install packages with: `pip install -r requirements.txt`
* Install the package itself: `run pip install .`

### Run/Train the models
* SOM-VAE: `cd /SOMVAE/somvae` -> `python somvae_train.py`
* DPSOM: `cd /DPSOM/dpsom` -> `python DPSOM.py`
* KMEANS: `cd /KMEANS` -> `python kmeans.py`
