// Contains all functions: move AND helper functions
#include <vector>
#include <iostream>

using namespace std;
using std::vector;

// Normalize and blur helper functions
vector< vector<float> > normalize(vector< vector <float> > grid) {
	float total = 0.0;
	int i;
	int j;
	vector<float> row;
	vector<float> newRow;
	float oldProb;
	for (i = 0; i < grid.size(); i++)
	{
		row = grid[i];
		// cout << "row size: " << row.size();
		for (j=0; j< row.size(); j++)
		{
			oldProb = row[j];
			total += oldProb;
		}
	}

	vector< vector<float> > newGrid;

	for (i = 0; i < grid.size(); i++) {
		vector<float> row = grid[i];
		newRow.clear();
		for (j=0; j< row.size(); j++) {
			float oldProb = row[j];
			float newProb = oldProb / total;
			newRow.push_back(newProb);
		}
		newGrid.push_back(newRow);
	}

	return newGrid;
}

vector < vector <float> > blur(vector < vector < float> > grid, float blurring) {

	vector < vector <float> > window;
	vector < vector <float> > newGrid;
	// vector < vector <float> > newGrid = vector(height, vector(width, 0.0) );
	vector <float> row;
	vector <float> newRow;

	int height;
	int width;
	float center, corner, adjacent;



	height = grid.size();
	width = grid[0].size();

	center = 1.0 - blurring;
	corner = blurring / 12.0;
	adjacent = blurring / 6.0;

	int i, j;
	float val;

	for (i=0; i<3; i++) {
		row.clear();
		for (j=0; j<3; j++) {
			switch (i) {
				case 0:
				switch (j) {
					case 0:
					val = corner;
					break;

					case 1:
					val = adjacent;
					break;

					case 2:
					val = corner;
					break;
				}
				break;

				case 1:
				switch (j) {
					case 0:
					val = adjacent;
					break;

					case 1:
					val = center;
					break;

					case 2:
					val = adjacent;
					break;
				}
				break;

				case 2:
				switch(j) {
					case 0:
					val = corner;
					break;

					case 1:
					val = adjacent;
					break;

					case 2:
					val = corner;
					break;
				}
				break;
			}
			row.push_back(val);
		}
		window.push_back(row);
	}


	vector <int> DX;
	vector <int> DY;

	DX.push_back(-1); DX.push_back(0); DX.push_back(1);
	DY.push_back(-1); DY.push_back(0); DY.push_back(1);

	int dx;
	int dy;
	int ii;
	int jj;
	int new_i;
	int new_j;
	float multiplier;
	float newVal;

	// initialize new grid to zeros
	for (i=0; i<height; i++) {
		newRow.clear();
		for (j=0; j<width; j++) {
			newRow.push_back(0.0);
		}
		newGrid.push_back(newRow);
	}


	for (i=0; i< height; i++ ) {
		for (j=0; j<width; j++ ) {
			val = grid[i][j];
			newVal = val;
			for (ii=0; ii<3; ii++) {
				dy = DY[ii];
				for (jj=0; jj<3; jj++) {
					dx = DX[jj];
					new_i = (i + dy + height) % height;
					new_j = (j + dx + width) % width;
					multiplier = window[ii][jj];
					newGrid[new_i][new_j] += newVal * multiplier;
				}
			}
		}
	}

	return normalize(newGrid);
}

// Move function defined below -- do not change
vector< vector <float> > move(int dy, int dx,
	vector < vector <float> > beliefs,
	float blurring)
{
	int height, width;
	height = beliefs.size();
	width = beliefs[0].size();

	float belief;
	vector < vector <float> > newGrid;

	// initialize new grid to zeros
	for (int i=0; i<height; i++) {
	    vector <float> newRow;
		newRow.clear();
		for (int j=0; j<width; j++) {
			newRow.push_back(0.0);
		}
		newGrid.push_back(newRow);
	}

	int new_i, new_j;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			new_i = (i + dy + height) % height;
			new_j = (j + dx + width)  % width;
			belief = beliefs[i][j];

			newGrid[new_i][new_j] = belief;
		}
	}
	return blur(newGrid, blurring);
}
