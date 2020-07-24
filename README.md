# trajectory-prediction-for-KalmanPrediction-and-DeepLearning

working on it....  

## Requirements:
* Python 3.x (I confirm that script run in Python 3.5.3)
* pytorch
* matplotlib (for visualizing results)

## Descrition: 
This repository is for studying a trajectory prediction using Kalman filter and deep learning models. 
Curretly, below **models** are implemented for trajectory-prediction.  

* **Kalman Filter model (KF)**: Using prediction and correction step in KF, update covariance matrix. At the trajectory-prediction step, it uses **Constant Velocity** or **Constant Acceleration** model.
* **Sequence-to-Sequence model based on GRU cell**  

## Dataset:
* [**Apolloscape**](http://apolloscape.auto/trajectory.html)  
: Apolloscape dataset is collected under various lighting conditions and traffic densities in Beijing, China. In this repo., we have used 'prediction_train.zip' file.
Each line in the file contains (frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading).

* [**Lyft**](https://self-driving.lyft.com/level5/data/)  
: Lyft data is collected under various urban conditions for 1000 hours driving. The dataset has 'zarr' format. For preprocessing data on your shoes, please check [this repository](https://github.com/zarr-developers/zarr-python).  

## futher detail of this repo.
* **main.py**: main fuction for running whole scripts. You can change hyper-parameters, dataset and a model.
* **model.py**:  
* **kalman_model.py**:  
* **vanilla_gru.py**:  
* **utils.py**:  

### References.
* http://apolloscape.auto/trajectory.html
* https://self-driving.lyft.com/level5/data/
* https://arxiv.org/pdf/1908.11472.pdf  


**Please ask a [Assistant](mailto:msk930512@snu.ac.kr "email address") if you have any questions.**

