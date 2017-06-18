#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;
  n_lidar_=2;
  n_radar_=3;
  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  double NIS_rad_;
  double NIS_las_;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.7;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;//0.3

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;//0.03

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // timestamp
  previous_timestamp_ = 0;
  //define spreading parameter
  lambda_ = 3 - n_aug_;
  // set weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0.0);
  double weight_0_ = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0_;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  return;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

   if (!is_initialized_) {
    // First Measurement
    x_ .fill(0.0);
    P_<< 1,0,0,0,0,
         0,1,0,0,0,
         0,0,1,0,0,
         0,0,0,1,0,
         0,0,0,0,1;


    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_(0);
      float theta= meas_package.raw_measurements_(1);
      float rho_dot= meas_package.raw_measurements_(2);
      float v = sqrt(pow(rho_dot*cos(theta),2)+pow(rho_dot*sin(theta),2));
      x_ << rho*cos(theta),rho*sin(theta),v,0.0,0.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {


      x_<< meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),0.0,0.0,0.0;
    }

    // done initializing, no need to predict or update


    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

   double delta_t =  (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

   Prediction(delta_t);

   if ( meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ )
   {
        UpdateRadar(meas_package);
   }
   else if ( meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ )
   {
        UpdateLidar(meas_package);

   }
   return;

}
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);
  Xsig_pred_.fill(0.0);

  //Predict Mean and Covariance
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);

  MatrixXd P_pred = MatrixXd(n_x_,n_x_);
  P_pred.fill(0.0);
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred.fill(0.0);

  AugmentedSigmaPoints(Xsig_aug);
  //std::cout << "AugmentedSigmaPoint" << std::endl;
  //std::cout << Xsig_aug << std::endl;

  SigmaPointPrediction(Xsig_pred, Xsig_aug,delta_t);

  //std::cout << "PredictedSigmaPoint" << std::endl;
  //std::cout << Xsig_pred << std::endl;

  PredictMeanAndCovariance(x_pred,P_pred,Xsig_pred);
  //std::cout << "Predicted Mean and Covariance" << std::endl;
  //std::cout << x_pred << std::endl;
  //std::cout << P_pred << std::endl;

  Xsig_pred_ = Xsig_pred;
  x_ = x_pred;
  P_ = P_pred;

  return;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */

void UKF::UpdateLidar(MeasurementPackage meas_package) {

  MatrixXd S = MatrixXd(n_lidar_,n_lidar_);
  S.fill(0.0);
  VectorXd z_pred = VectorXd(n_lidar_);
  z_pred.fill(0.0);
  MatrixXd T = MatrixXd(n_x_, n_lidar_);
  T.fill(0.0);

  PredictLidarMeasurement(z_pred,T,S);
  UpdateStateLidar(meas_package,z_pred,T,S);

  previous_timestamp_ = meas_package.timestamp_;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  MatrixXd S = MatrixXd(n_radar_,n_radar_);
  S.fill(0.0);
  VectorXd z_pred = VectorXd(n_radar_);
  z_pred.fill(0.0);

  MatrixXd T = MatrixXd(n_x_, n_radar_);
  T.fill(0.0);

  PredictRadarMeasurement(z_pred,T,S);
  UpdateStateRadar(meas_package,z_pred,T,S);
  previous_timestamp_ = meas_package.timestamp_;

}



void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);
  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  //std::cout << "L" << std::endl;

  //std::cout << L << std::endl;

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  Xsig_out = Xsig_aug;
  return;
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_out, MatrixXd& Xsig_aug, double delta_t) {
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred.fill(0.0);
  //double delta_t = 0.1; //time diff in sec

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }


    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;






    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //std::cout << "delta_t" << std::endl;
    //std::cout << delta_t << std::endl;
    //std::cout << "yawd" << std::endl;
    //std::cout << yawd << std::endl;
    //std::cout << "v" << std::endl;
    //std::cout << v << std::endl;


    //std::cout << "yawd" << std::endl;
    //std::cout << yawd << std::endl;

    //std::cout << "px_p" << std::endl;

    //std::cout << px_p << std::endl;
    //std::cout << "py_p" << std::endl;

    //std::cout << py_p << std::endl;

    //std::cout << "v_p" << std::endl;
    //std::cout << v_p << std::endl;


    //std::cout << "v_p" << std::endl;
    //std::cout << v_p << std::endl;

    //std::cout << "yaw_p" << std::endl;
    //std::cout << yaw_p << std::endl;

    //std::cout << "yawd_p" << std::endl;
    //std::cout << yawd_p << std::endl;


    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }


  //print result
  //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  Xsig_out = Xsig_pred;
  return;
}

void UKF::PredictMeanAndCovariance(VectorXd& x_out, MatrixXd& P_out,MatrixXd &Xsig_pred) {

  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_+ 1; i++) {  //iterate over sigma points
     x = x+ weights_(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //write result
  x_out = x;
  P_out = P;
  return;
}

void UKF::PredictRadarMeasurement(VectorXd& z_out, MatrixXd& Tc_out ,MatrixXd& S_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_radar_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_radar_,n_radar_);
  S.fill(0.0);
  // Create Matrix
  MatrixXd Tc = MatrixXd(n_x_,n_radar_);
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_+ 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_radar_,n_radar_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;


  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  z_out = z_pred;
  S_out = S;
  Tc_out = Tc;
  return;
}

void UKF::UpdateStateRadar(MeasurementPackage meas_package,VectorXd& z_pred, MatrixXd& Tc, MatrixXd& S) {


  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_radar_);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),meas_package.raw_measurements_(2);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_rad_ = z_diff.transpose() * S.inverse() * z_diff;
  return;
}


void UKF::PredictLidarMeasurement(VectorXd& z_out, MatrixXd& Tc_out ,MatrixXd& S_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_lidar_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_+ 1; i++) {  //2n+1 simga points

    // measurement model
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) =  Xsig_pred_(1,i);

  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_lidar_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_lidar_,n_lidar_);
  S.fill(0.0);
  // Create Matrix
  MatrixXd Tc = MatrixXd(n_x_, n_lidar_);
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_lidar_,n_lidar_);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S = S + R;


  //write result
  z_out = z_pred;
  S_out = S;
  Tc_out = Tc;
  return;

}




void UKF::UpdateStateLidar(MeasurementPackage meas_package,VectorXd& z_pred, MatrixXd& Tc, MatrixXd& S) {


  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_lidar_);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_las_ = z_diff.transpose() * S.inverse() * z_diff;
  return;


}



