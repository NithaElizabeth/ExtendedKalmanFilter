# Extended Kalman filter algorithm for mobile robot localization
Localization of mobile robots plays a vital role in trying to comprehend the behaviour of an autonomous robot, where the robot must invariably discern its position while moving in a given map. The main issue of the mobile robot localization is that the mobile robot must continuously affirm its location in order to successfully accomplish its given task. With the growing stipulation for robot localization, the Extended Kalman Filter (EKF) algorithm has received copious attention. 
The EKF algorithm is more realistic in non-linear systems as most systems are non-linear in the field of autonomous engineering, which has an autonomous white noise in both the system and the estimation model.

![Figure_1](https://user-images.githubusercontent.com/47361086/126139029-70769bd1-f440-427b-b9ce-89614aab5589.png)

The figure above depicts the sensor fusion localization with Extended Kalman Filter(EKF). The blue line is true trajectory, the black line is dead reckoning trajectory, the magenta points are positioning observations, and the yellow line is estimated trajectory with EKF. The turqoise ellipse is estimated covariance ellipse with EKF.
