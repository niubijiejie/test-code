#include <iostream>

using namespace std;

int main()
{
    float a[3];
    float *p;
    float b[3]={1,2,3};
    float *p1;
    p1=b;
    //p1=p;
    //a=p1;
    //p1=a;
    //&a[0]=p1;
    for(int i=0;i<3;i++)
    {
        a[i]=*(p1+i);
    }
    cout << a[0] << endl;
    cout << a[1] << endl;
    cout << a[2] << endl;
    return 0;
}

