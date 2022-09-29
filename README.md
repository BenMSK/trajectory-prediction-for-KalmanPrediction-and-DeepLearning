# trajectory-prediction-for-KalmanPrediction-and-DeepLearning
There is no further plan for developing some projects from this repository, however, for studying the trajectory prediction research, this working code would be a good start.  

## Requirements:
* Python 3.x (I confirm that the scripts run in Python 3.5.3)
* pytorch
* matplotlib (for visualizing results)

## Description: 
This repository is for studying a trajectory prediction using Kalman filter and deep learning models. 
There are some confused parts on the script, but this code will be a good start for the trajectory prediction study.
Curretly, below **models** are implemented for trajectory-prediction.  

* **Kalman Filter model (KF)**: Using prediction and correction step in KF, update covariance matrix. At the trajectory-prediction step, it uses **Constant Velocity** or **Constant Acceleration** model.
* **Sequence-to-Sequence model based on GRU cell**  

## Dataset:
* [**Apolloscape**](http://apolloscape.auto/trajectory.html)  
: Apolloscape dataset is collected under various lighting conditions and traffic densities in Beijing, China. In this repo., we have used 'prediction_train.zip' file.
Each line in the file contains (frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading).

* [**Lyft**](https://self-driving.lyft.com/level5/data/)  
: Lyft data is collected under various urban conditions for 1000 hours driving. The dataset has 'zarr' format. For preprocessing data on your shoes, please check [this repository](https://github.com/zarr-developers/zarr-python).  

## Further Details of This Repository.
* **main.py**: main fuction for running whole scripts. You can change hyper-parameters, dataset and a model.
* **model.py**: Training or testing a model
* **kalman_model.py**: a basic kalman filter model for trajectory prediction. It uses the prediction only. However, as the prediction step is on-going, the covariance is getting larger as the kalman filter does. As a result, you can sample the trajectory prediction results from Kalman model.
* **vanilla_gru.py**: a deep-learning model for learning the trajectory sequence. Currently, it consists of a Encoder-Decoder architecture.
* **utils.py**: some utitlity functions, mainly preprocessing data. 

### References.
* http://apolloscape.auto/trajectory.html
* https://self-driving.lyft.com/level5/data/
* https://arxiv.org/pdf/1908.11472.pdf  


**Please contact me (msk930512@snu.ac.kr) if you have any questions.**

