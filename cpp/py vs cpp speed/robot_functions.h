// This is the header file, which includes function declarations
#include <vector>

// Declare the move function - its return type and parameters
std::vector< std::vector <float> > move(int dy, int dx,
	std::vector < std::vector <float> > beliefs,
	float blurring);

// Declaring the blur function
std::vector < std::vector <float> > blur(std::vector < std::vector < float> > grid,
    float blurring);
