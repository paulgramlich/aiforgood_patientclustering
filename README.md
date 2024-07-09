
# AI FOR GOOD: Patient clustering with VAE-SOM AND DPSOM 
### Ludwig-Maximilians-Universität München
### Felix Steckenbiller & Paul Gramlich

This project contains a new approach to patient clustering. 

### Dataset 

`/Data/preprocessing.py`
We used a Lower Back Pain (LBP) patient dataset, that was preprocessed by us to convert categorical variables into numerical values. 
Furthermore, NAs were removed and replaced with `0` and scaled the data from `[0,1]`.
An scaled and unscaled version for plotting can be found in `/Data/`.

`/Data/processing.py`
Then the dataset was processed to fit the requirements of the models. 
The provided script is just an excerpt of the code that was then inserted injected into the training files of the models.

### Models
We used 2 models and one baseline model: 
* VAE-SOM (original model and paper: https://github.com/ratschlab/SOM-VAE) `/VAESOM`
* DPSOM (original model and paper: https://github.com/ratschlab/dpsom) `/DPSOM`
* Baseline Model: K-MEANS++ `/KMEANS`

### Technical requirements
Technical requirements for each model can be found in the README file in each subdirectory.
