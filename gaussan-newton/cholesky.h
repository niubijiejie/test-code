#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;

namespace TooN {
class Cholesky
{
private:
        //std::vector<float> cholesky;
        //int size, rank;
        Mat mycholesky;
        int rank;

public:
        Cholesky( Mat &m);

        ~Cholesky(void);

        Mat Backsub( Mat &m1) const;
};
}
