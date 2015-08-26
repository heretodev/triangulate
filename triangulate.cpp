#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include "triangulate.h"

Eigen::MatrixXd cameraMatrixKnownRT(const double rx, const double ry, const double rz, const double tx, const double ty, const double tz)
{
	// Rotation matrix R = Rx * Ry * Rz(Euler's rotation theorem: any rotation can be broken into 3 angles)
	// Rx rotation by rx about x:
	Eigen::MatrixXd Rx(3,3), Ry(3,3), Rz(3,3);
	Rx << cos(rx), -sin(rx), 0,
		  sin(rx), cos(rx), 0,
		  0, 0, 1;

	// Ry rotation by ry about y:
	Ry << cos(ry), 0, -sin(ry),
		  0, 1, 0,
		  sin(ry), 0, cos(ry);

	// Rz rotation by rz about z:
	Rz << cos(rz), sin(rz), 0,
		  -sin(rz), cos(rz), 0,
		  0, 0, 1;

	Eigen::MatrixXd R = Rx * Ry * Rz;

	Eigen::MatrixXd P(3,4);
	P << R(0, 0), R(0, 1), R(0, 2), tx,
		 R(1, 0), R(1, 1), R(1, 2), ty,
		 R(2, 0), R(2, 1), R(2, 2), tz;
	return P;
}

/* With P1 set to origin, finds camera matrix P2 from the fundamental matrix */
template <typename Type>
void camerasFromFundamentalMatrix(cv::Mat F, cv::Mat& P1, cv::Mat& P2)
{
	// Set P1 to origin
	P1 = cv::Mat::eye(3, 4, (sizeof(Type) == sizeof(float)) ? CV_32FC1 : CV_64FC1);

	// Get epipole ep2 from F:
	cv::SVD FSVD = cv::SVD(F);
	cv::Mat ep2 = FSVD.u.col(2);

	// P2 = [ep2_x' * E, ep2]:
	// The skew - symmetric matrix of ep2
	cv::Mat ep2_x = (cv::Mat_<Type>(3, 3) <<
		0, -ep2.at<Type>(2), ep2.at<Type>(1),
		ep2.at<Type>(2), 0, -ep2.at<Type>(0),
		-ep2.at<Type>(1), ep2.at<Type>(0), 0);
	P2 = ep2_x.t() * F;
	cv::hconcat(P2, ep2, P2);
}

// 12.2 (Hartley & Zisserman) : Use DLT : From x1(cross) P1 * X = 0
// i.e.x1 cross x1 = 0, where x1 = P1 * X.
// x1 is 2xN, X is Nx3
Eigen::MatrixXd triangulateCorrespondences(Eigen::MatrixXd& x1, Eigen::MatrixXd& x2, Eigen::MatrixXd& P1, Eigen::MatrixXd& P2)
{
	Eigen::MatrixXd X(x1.cols(), 3);
	Eigen::MatrixXd A(4, 4);
	for (unsigned int i = 0; i < x1.cols(); i++)
	{
		A.row(0) = x1(0, i) * P1.row(2) - P1.row(0);
		A.row(1) = x1(1, i) * P1.row(2) - P1.row(1);
		A.row(2) = x2(0, i) * P2.row(2) - P2.row(0);
		A.row(3) = x2(1, i) * P2.row(2) - P2.row(1);
		// Solve for X, AX = 0, last singular value is 0
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::MatrixXd V = svd.matrixV();
		if (abs(V(3, 3)) < .001)
			V(3, 3) = 1.0;
		//X.row(i) = (V.col(3) / V(3, 3)).transpose();
		X(i, 0) = V(0, 3) / V(3, 3);
		X(i, 1) = V(1, 3) / V(3, 3);
		X(i, 2) = V(2, 3) / V(3, 3);
	}
	return X;
}

Eigen::MatrixXd ptcloudFromCorrespondences(const std::string& correspondencesFilename, const std::string& extrinsicsFilename, const std::string& ptcloudFilename)
{
	// THE UNDISTORTED POINTS (calibrated by K in P1, P2)
	std::ifstream in1(correspondencesFilename);
	if (in1.bad())
		return Eigen::MatrixXd(1,3);
	unsigned int N = 0;
	char buf[256];
	for (; in1.good(); N++)
		in1.getline(buf, 256);
	N--; // Advanced past for loop
	in1.close();

	std::ifstream in(correspondencesFilename);
	if (in.bad())
		return Eigen::MatrixXd(1, 3);
	char c;
	std::vector< cv::Point2d > x1v, x2v;
	for (unsigned int i = 0; i < N && !in.bad(); i++)
	{
		double x1x, x1y, x2x, x2y, max, qwx, qwy;
		in >> x1x >> c >> x1y >> c >> x2x >> c >> x2y >> c >> max >> c >> qwx >> c >> qwy;
		// Hardcoded threshold of .6, qw > .5
		if ((max > .6))//  && (qwx > .5) && (qwy > .5)) // if stabilization
		{
			x1v.push_back(cv::Point2d(x1x, x1y));
			x2v.push_back(cv::Point2d(x2x, x2y));
		}
	}
	Eigen::MatrixXd x1(2, x1v.size());
	Eigen::MatrixXd x2(2, x2v.size());
	unsigned int co = 0;
	for (std::vector<cv::Point2d>::iterator x1r = x1v.begin(), x2r = x2v.begin(); ((x1r != x1v.end()) && (x2r != x2v.end())); x1r++, x2r++, co++)
	{
		x1(0, co) = x1r->x;
		x1(1, co) = x1r->y;
		x2(0, co) = x2r->x;
		x2(1, co) = x2r->y;
	}

	Eigen::MatrixXd P1 = Eigen::MatrixXd::Identity(3, 4), P2 = Eigen::MatrixXd::Identity(3, 4);
	P2(0, 3) = 1.0;
	if (extrinsicsFilename != "")
	{
		std::ifstream extrinsicsFile(extrinsicsFilename);
		if (extrinsicsFile.bad())
			return Eigen::MatrixXd(1, 3);
		double rx, ry, rz, tx, ty, tz;
		extrinsicsFile >> rx >> c >> ry >> c >> rz >> c;
		extrinsicsFile >> tx >> c >> ty >> c >> tz >> c;
		std::cout << "Getting extrinsics: " << rx << ',' << ry << ',' << rz << ',' << tx << ',' << ty << ',' << tz << std::endl;
		P2 = cameraMatrixKnownRT(rx, ry, rz, tx, ty, tz);
	}
	Eigen::MatrixXd KR(3, 3), KL(3, 3);
	KR << 822.81256, 0, 958.56934,
		0, 822.31793, 538.86395,
		0, 0, 1.0;
	KL << 811.90875, 0, 958.48077,
		0, 811.38806, 538.83582,
		0, 0, 1.0;
	P1 = KL * P1;
	P2 = KL * P2; // Only left for box dataset!
	std::cout << "P1: " << P1 << std::endl << "P2: " << P2 << std::endl;
	Eigen::MatrixXd X = triangulateCorrespondences(x1, x2, P1, P2);
	std::ofstream out(ptcloudFilename);
	const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
	out << X.format(CSVFormat);
	out.close();

	return X;
}

/* With P1 set to origin, finds camera matrix P2 from the essential matrix, resolving the four-fold ambiguity described in 9.6.2 of Hartley & Zisserman
   Requires floating point test image correspondence., leftCorrespondences is 2xN
template <typename Type>
void camerasFromEssentialMatrix(cv::Mat E, cv::Mat& P1, cv::Mat& P2, cv::Mat& leftCorrespondences, cv::Mat& rightCorrespondences)
{
	// Set P1 to origin
	P1 = cv::Mat::eye(3, 4, (sizeof(Type) == sizeof(float)) ? CV_32FC1 : CV_64FC1);

	// 4 solutions: Result 9.19 Hartley and Zisserman.
	cv::SVD ESVD = cv::SVD(E);
	cv::Mat s1(3, 4, (sizeof(Type) == sizeof(float)) ? CV_32FC1 : CV_64FC1), s3(3, 4, (sizeof(Type) == sizeof(float)) ? CV_32FC1 : CV_64FC1);
	cv::Mat s1_2 = ESVD.u * cv::Mat::diag(ESVD.w) * ESVD.vt; // E itself
	s1.at<Type>(0, 0) = s1_2.at<Type>(0, 0);
	s1.at<Type>(0, 1) = s1_2.at<Type>(0, 1);
	s1.at<Type>(0, 2) = s1_2.at<Type>(0, 2);
	s1.at<Type>(1, 0) = s1_2.at<Type>(1, 0);
	s1.at<Type>(1, 1) = s1_2.at<Type>(1, 1);
	s1.at<Type>(1, 2) = s1_2.at<Type>(1, 2);
	s1.at<Type>(2, 0) = s1_2.at<Type>(2, 0);
	s1.at<Type>(2, 1) = s1_2.at<Type>(2, 1);
	s1.at<Type>(2, 2) = s1_2.at<Type>(2, 2);
	s1.col(3) = ESVD.u.row(2).t();
	cv::Mat s2;
	s1.copyTo(s2); // last col is negative
	s2.col(3) = -s2.col(3);
	cv::Mat s3_4 = ESVD.u * cv::Mat::diag(ESVD.w).t() * ESVD.vt;
	s3.at<Type>(0, 0) = s3_4.at<Type>(0, 0);
	s3.at<Type>(0, 1) = s3_4.at<Type>(0, 1);
	s3.at<Type>(0, 2) = s3_4.at<Type>(0, 2);
	s3.at<Type>(1, 0) = s3_4.at<Type>(1, 0);
	s3.at<Type>(1, 1) = s3_4.at<Type>(1, 1);
	s3.at<Type>(1, 2) = s3_4.at<Type>(1, 2);
	s3.at<Type>(2, 0) = s3_4.at<Type>(2, 0);
	s3.at<Type>(2, 1) = s3_4.at<Type>(2, 1);
	s3.at<Type>(2, 2) = s3_4.at<Type>(2, 2);
	s3.col(3) = ESVD.u.row(2).t();
	cv::Mat s4;
	s3.copyTo(s4); // last col is negative
	s4.col(3) = -s4.col(3);

	// 9.6.3 Resolve 4 - fold ambiguity : linearly triangulate one point for each solution.
	// If that point has a positive depth(in front of both cameras), it's the correct solution.
	cv::Mat testPoint4D_1, testPoint4D_2, testPoint4D_3, testPoint4D_4;
	unsigned int ti = floor(rand() % leftCorrespondences.cols);

	testPoint4D_1 = triangulateCorrespondences(leftCorrespondences.col(ti), rightCorrespondences.col(ti), P1, s1);
	testPoint4D_2 = triangulateCorrespondences(leftCorrespondences.col(ti), rightCorrespondences.col(ti), P1, s2);
	testPoint4D_3 = triangulateCorrespondences(leftCorrespondences.col(ti), rightCorrespondences.col(ti), P1, s3);
	testPoint4D_4 = triangulateCorrespondences(leftCorrespondences.col(ti), rightCorrespondences.col(ti), P1, s4);

	// Now scale homogeneous component and test if it has positive depth.  Note: always float due to OpenCV's triangulatePoints
	if ((testPoint4D_1.at<Type>(2) / testPoint4D_1.at<Type>(3)) > 0)
		P2 = s1;
	else if ((testPoint4D_2.at<Type>(2) / testPoint4D_2.at<Type>(3)) > 0)
		P2 = s2;
	else if ((testPoint4D_3.at<Type>(2) / testPoint4D_3.at<Type>(3)) > 0)
		P2 = s3;
	else if ((testPoint4D_4.at<Type>(2) / testPoint4D_4.at<Type>(3)) > 0)
		P2 = s4;
	else
		P2 = P1; // default
}
*/