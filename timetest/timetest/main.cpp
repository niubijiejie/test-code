#include <iostream>
#include <chrono>

using namespace std;

int main()
{
    int64_t lastDepthTime = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    cout << "Hello World!" << endl;
    cout << lastDepthTime << endl;
    return 0;
}

