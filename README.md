# birdify
## _Classification of bird species based on vocalization audio recordings using Convolutional Neural Networks_
![Alt text](figures/parus_major.jpg?raw=true "Parus major")

## About
The problem of bird vocalization classification is well known and has been addressed
numerous times. Specifically in Data Science papers, it has appeared repeatedly over
the years with progressively better attempts. One of succesful approaches is using
Convolutional Neural Network with visual representation for audio signal
(e.g. (mel) spectrograms) as features. Such approach is also implemented in this project.

## Project structure
**Main project file** (Jupyter Notebook): ``` birdify.ipynb```\
Library containing all core preprocessing methods can be found under: ```/soundmatrix```\
Csv file containing information about samples: ```dataset.csv```\
Python script for resampling and signal/noise splitting: ```sample_gen.py```

## Dependencies
    pandas
    numpy
    librosa
    pydub
    scipy
    tensorflow
    tensorflow-io

## Dataset

All audio data used in this project come from **Bird songs from Europe (xeno-canto)**
collection built from a xeno-canto API query and uploaded on Kaggle by Francisco de Abreu e Lima.


---
## References

[1] Sprengel, E., Jaggi, M., Kilcher, Y., & Hofmann, T. (2016). Audio Based Bird Species Identification using Deep Learning Techniques. CLEF.\
[2] Kahl, S., Wilhelm-Stein, T., Hussein, H., Klinck, H., Kowerko, D., Ritter, M., & Eibl, M. (2017). Large-Scale Bird Sound Classification using Convolutional Neural Networks. CLEF.\
[3] Smith, L. N., (2017). Cyclical Learning Rates for Training Neural Networks. IEEE Winter Conference on Applications of Computer Vision.\
[4] Lasseck, M., (2013). Bird song classification in field recordings: winning solution for nips4b2013 competition. Proc. of int. symp. Neural Information Scaled for Bioacoustics.\
[5] Abreu e Lima, F., (2020). Bird songs from Europe (xeno-canto). https://www.kaggle.com/dsv/1029985