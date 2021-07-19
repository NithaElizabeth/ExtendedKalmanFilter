# Extended Kalman filter algorithm for mobile robot localization
Localization of mobile robots plays a vital role in trying to comprehend the behaviour of an autonomous robot, where the robot must invariably discern its position while moving in a given map. The main issue of the mobile robot localization is that the mobile robot must continuously affirm its location in order to successfully accomplish its given task. With the growing stipulation for robot localization, the Extended Kalman Filter (EKF) algorithm has received copious attention. 
The EKF algorithm is more realistic in non-linear systems as most systems are non-linear in the field of autonomous engineering, which has an autonomous white noise in both the system and the estimation model.

![Figure_1m](https://user-images.githubusercontent.com/47361086/126142143-2130392b-f108-458b-81bc-f1b6094dfa4e.png)

The figure above depicts the sensor fusion localization with Extended Kalman Filter(EKF). The blue line is true trajectory, the black line is dead reckoning trajectory, the magenta points are positioning observations, and the yellow line is estimated trajectory with EKF. The turqoise ellipse is estimated covariance ellipse with EKF.
## Installation and Running Procedure
Clone this github repository into the workspace
```
git clone https://github.com/NithaElizabeth/ExtendedKalmanFilter
```
Next install all the necessary libraries .
```
pip install -r requirements.txt
```
After this launch the program
```
python EKF.py

## Author
The system was developed by Nitha Elizabeth John.
Author  : Nitha Elizabeth John\
Contact : nithaelizabethjohn@gmail.com
