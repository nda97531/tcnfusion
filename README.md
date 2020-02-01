#### 1. Overview

This code implements a TCN-based fusion model used for human activity recognition.
The model takes in 2 input streams, which are ***acceleration*** and ***skeleton***, and gives a classification result of a 3-second-length window.

- *data_processing/noise_filtering.py*: Lowpass filter for acceleration data.<br/>
- *data_processing/preprocess_skeleton.py*: Select joints and add angle features to skeleton data.<br/>
- *model_and_dataset/keras_model_tcn.py*: TCN for single-modal data.<br/>
- *model_and_dataset/keras_model_fusion.py*: Fusion for multi-modal features.<br/>
- *model_and_dataset/keras_model_endtoend.py*: Complete fusion model.<br/>

#### 2. Dataset

The model is built and evaluated on CMDFALL, a multi-modal dataset for HAR.<br />
Original paper of this dataset: [A multi-modal multi-view dataset for human fall analysis and preliminary investigation on modality](https://ieeexplore.ieee.org/document/8546308)<br/>
This dataset is available at: [https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html](https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html) <br />

#### 3. Dependencies
 - Python
 - Numpy
 - Scipy
 - Tensorflow 1.14
 - [Keras-rectified-adam](https://pypi.org/project/keras-rectified-adam/)