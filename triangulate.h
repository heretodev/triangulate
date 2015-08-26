#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

// 12.2 (Hartley & Zisserman) : Use DLT : From x1(cross) P1 * X = 0
// i.e.x1 cross x1 = 0, where x1 = P1 * X.
// x1 is 2xN, X is Nx3
Eigen::MatrixXd triangulateCorrespondences(Eigen::MatrixXd& x1, Eigen::MatrixXd& x2, Eigen::MatrixXd& P1, Eigen::MatrixXd& P2);

Eigen::MatrixXd ptcloudFromCorrespondences(const std::string& correspondencesFilename, const std::string& extrinsicsFilename, const std::string& ptcloudFilename);