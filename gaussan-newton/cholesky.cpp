#include "cholesky.h"
#include <iostream>
using namespace TooN;

Cholesky::Cholesky( Mat &m)
{
    mycholesky=m;
    int size = mycholesky.rows;
    for(int j=0;j<size;j++)
    {
        double inv_diag = 1;
        double *p;
        for(int i=j;i<size;i++)
        {
            p = mycholesky.ptr<double>(i);
            double val = p[j];
            double *p1,*p2;
            p2 = mycholesky.ptr<double>(i);
            for(int j2=0;j2<j;j2++)
            {
                p1 = mycholesky.ptr<double>(j2);

                val -= p1[j]*p2[j2];
            }
            if(j==i)
            {
                p[j]=val;
                if(val==0)
                {
                    rank = i;
                    return;
                }
                inv_diag=1/val;
            }
            else
            {
                p1 = mycholesky.ptr<double>(j);
                p1[i] = val;
                p[j] = val*inv_diag;
            }
        }
    }
    rank =size;
    /*for(int i=0;i<mycholesky.rows;i++)
    {
        double *p;
        p=mycholesky.ptr<double>(i);
        for(int j=0;j<mycholesky.cols;j++)
        {
            std::cout<<p[j]<<endl;
        }
        std::cout<<endl;
    }*/
}


Mat Cholesky::Backsub( Mat &m1) const
{
    //Mat m1(m);
    int size = mycholesky.rows;
    Mat y(size, m1.cols, CV_64F);
    for(int i=0;i<size;i++)
    {
        double *p3,*p6;
        p3 = m1.ptr<double>(i);
        p6 = y.ptr<double>(i);
        double val=p3[0];
        const double *p4;
        double *p5;
        p4 = mycholesky.ptr<double>(i);
        for(int j=0;j<i;j++)
        {

            p5 = y.ptr<double>(j);
            val -= p4[j]*p5[0];
        }
        p6[0]=val;
    }

    for(int i=0;i<size;i++)
    {
        const double *p6;
        double *p7;
        p6 = mycholesky.ptr<double>(i);
        p7 = y.ptr<double>(i);
        p7[0] /= p6[i];
    }

    Mat result(size, m1.cols, CV_64F);
    for(int i=size-1;i>=0;i--)
    {
        const double *p3;
        double *p4,*p5,*p6;
        p4 = result.ptr<double>(i);
        p6 = y.ptr<double>(i);
        double val=p6[0];
        for(int j=i+1;j<size;j++)
        {
            p3 = mycholesky.ptr<double>(j);
            p5 = result.ptr<double>(j);
            val -= p3[i]*p5[0];
        }
        p4[0]=val;
    }
    /*for(int i=0;i<size;i++)
    {
        double *p3;
        p3 = result.ptr<double>(i);
        p3[0] = p3[0]*p3[0];
     }*/
   /* for(int i=0;i<result.rows;i++)
    {
        double *p;
        p=result.ptr<double>(i);
        for(int j=0;j<result.cols;j++)
        {
            std::cout<<p[j]<<endl;
        }
        std::cout<<endl;
    }*/
     return result;
}


Cholesky::~Cholesky(void)
{
}

