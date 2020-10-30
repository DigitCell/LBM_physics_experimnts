#include <iostream>
#include "mainloop.hpp"

using namespace std;

int main()
{
    cout << "Hello Cephalopods!" << endl;
    MainLoop mainloop;
    mainloop.Run();
    return 0;
}
