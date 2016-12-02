//一个简单的高斯牛顿法示例

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "cholesky.h"
#include <iostream>

using namespace std;
using namespace cv;

const double DERIV_STEP = 1e-5;
const int MAX_ITER = 100;


void GaussNewton(double(*Func)(const Mat &input, const Mat &params), // 主要函数
				 const Mat &inputs, const Mat &outputs, Mat &params);

double Deriv(double(*Func)(const Mat &input, const Mat &params), // 求导函数
			 const Mat &input, const Mat &params, int n);

// 目标函数定义
double Func(const Mat &input, const Mat &params);

int main()
{

	// F = A*sin(Bx) + C*cos(Dx)
    // 4个参数: A, B, C, D
	int num_params = 4;

    // 总的迭代次数
    int total_data = 100;
    //4个参数实际的值
    double A = 5;
    double B = 1;
    double C = 10;
    double D = 2;

    Mat inputs(total_data, 1, CV_64F);
    Mat outputs(total_data, 1, CV_64F);

    for(int i=0; i < total_data; i++) {
        double x = -10.0 + 20.0* rand() / (1.0 + RAND_MAX); // 产生随机数 [-10 到 10]
        double y = A*sin(B*x) + C*cos(D*x);

        //观测变量代入

        inputs.at<double>(i,0) = x;
        outputs.at<double>(i,0) = y;
    }

    // 设定合理初值
    Mat params(num_params, 1, CV_64F);

    params.at<double>(0,0) = 1;
    params.at<double>(1,0) = 1;
    params.at<double>(2,0) = 3; //如果这里设置成1初值太远
    params.at<double>(3,0) = 1.8;

    GaussNewton(Func, inputs, outputs, params);

    printf("True parameters: %f %f %f %f\n", A, B, C, D);
    printf("Parameters from GaussNewton: %f %f %f %f\n", params.at<double>(0,0), params.at<double>(1,0),
    													params.at<double>(2,0), params.at<double>(3,0));

    /*Mat ATA = (Mat_<double>(2,2)<<2,2,2,4);
    Mat ATb = (Mat_<double>(2,1)<<6,10);
    TooN::Cholesky cholA(ATA);

   // cout << cholA.mycholesky<<endl<<endl;
    Mat delta=cholA.Backsub(ATb);
    //cout << delta <<endl<<endl;*/
    return 0;
}

double Func(const Mat &input, const Mat &params)
{
    //得到目标函数

	double A = params.at<double>(0,0);
	double B = params.at<double>(1,0);
	double C = params.at<double>(2,0);
	double D = params.at<double>(3,0);

	double x = input.at<double>(0,0);

    return A*sin(B*x) + C*cos(D*x);
}

double Deriv(double(*Func)(const Mat &input, const Mat &params), const Mat &input, const Mat &params, int n)
{

    // 返回导数值
	Mat params1 = params.clone();
	Mat params2 = params.clone();

    // 使用中心差分计算导数
	params1.at<double>(n,0) -= DERIV_STEP;
	params2.at<double>(n,0) += DERIV_STEP;

	double p1 = Func(input, params1);
	double p2 = Func(input, params2);

	double d = (p2 - p1) / (2*DERIV_STEP);

	return d;
}

void GaussNewton(double(*Func)(const Mat &input, const Mat &params),
				 const Mat &inputs, const Mat &outputs, Mat &params)
{
    int m = inputs.rows;
    int n = inputs.cols;
    int num_params = params.rows;

    Mat r(m, 1, CV_64F); // 残差矩阵
    Mat Jf(m, num_params, CV_64F); // 约旦矩阵
    Mat input(1, n, CV_64F);

    double last_mse = 0;

    for(int i=0; i < MAX_ITER; i++) {
        double mse = 0;

        for(int j=0; j < m; j++) {
        	for(int k=0; k < n; k++) {
        		input.at<double>(0,k) = inputs.at<double>(j,k);
        	}

            r.at<double>(j,0) = outputs.at<double>(j,0) - Func(input, params);

            mse += r.at<double>(j,0)*r.at<double>(j,0);

            for(int k=0; k < num_params; k++) {
            	Jf.at<double>(j,k) = Deriv(Func, input, params, k);
            }
        }

        mse /= m;

        // 阈值判断
        if(fabs(mse - last_mse) < 1e-8) {
        	break;
        }
        Mat ATA = Jf.t()*Jf;
        Mat ATb = Jf.t()*r;
        TooN::Cholesky cholA(ATA);
        Mat delta=cholA.Backsub(ATb);
        //Mat delta = ((Jf.t()*Jf)).inv() * Jf.t()*r;
       /* for(int i1=0;i1<delta.rows;i1++)
        {
            double *p;
            p=delta.ptr<double>(i1);
            for(int j=0;j<delta.cols;j++)
            {
                std::cout<<p[j]<<endl;
            }
            std::cout<<endl;
        }*/
        params += delta;

        //均方差差输出
        printf("%d %f\n", i, mse);

        last_mse = mse;
    }
}
