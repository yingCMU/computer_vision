#include <iostream>
#include "robot_functions.h"  // including the header file

using namespace std;
using std::vector;

// Main function
int main() {

    /* Defining the initial beliefs grid:
     [ [0.05f, 0.2f, 0.2f, 0.05f, 0.05f],
       [0.05f, 0.1f, 0.2f, 0.05f, 0.05f] ]
       */
    vector< vector< float > > beliefs_grid;

    // Constructing the row vectors (from arrays)
    float a_row1[] = {0.05f, 0.2f, 0.2f, 0.05f, 0.05f};
    vector<float> vec_row1 (a_row1, a_row1 + sizeof(a_row1) / sizeof(a_row1[0]) );
    float a_row2[] = {0.05f, 0.1f, 0.2f, 0.05f, 0.05f};
    vector<float> vec_row2 (a_row2, a_row2 + sizeof(a_row2) / sizeof(a_row2[0]) );
    // Adding vectors to the beliefs grid
    beliefs_grid.push_back(vec_row1);
    beliefs_grid.push_back(vec_row2);

    /* Timing the move function!
     Start the timer below:
     */
    int start_s=clock();
	// Run move 1000 times - move right by 2 each iteration
	// TODO: Change the number of iterations and notice the change in time
	int iterations = 1000;

	for(int i = 0; i < iterations; i++){
	    move(0, 2, beliefs_grid, 1.0f);
	}
	// Stop the timer and display the result!
    int stop_s=clock();

    cout << "Time to move "<<iterations<<" iterations in milliseconds: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;

    return 0;
}
