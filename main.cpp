
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <fstream>
#include <vector>
#include <iostream>
#include <math.h>
#include <armadillo>
#include <cmath>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace arma;

string inputf;
unsigned char* board;
unsigned char* board2;

struct V { double x; double y; };
struct rgb { int r; int g; int b; };
int isize_x, isize_y;

// PART 1 variables ========================================================================
// I use mat variable type provided by Amarillo to keep the vector components
mat X;
mat Y;
mat XX(512, 512); 
mat YY(512, 512);
mat NOISE(512, 512);

// PART 2 variables ========================================================================
// to be used for time dependant vector fields :
// before interpolation
vector <mat> timed_X_raw;
vector <mat> timed_Y_raw;
// after interpolation
vector <mat> Xi;
vector <mat> Yi;
vector <mat> lics;
vector <vector <V>> streamlines;
vector <vector <V>> streaks;

int x_size, y_size, t_size;
int frame;

// Function to convert Hue and Value to RGB
rgb hsv2rgb(double h, double v) {
	double s = 1;
	int hi = floor(h / 60.0);
	double f = h / 60.0 - hi;
	double p = v*(1 - s);
	double q = v*(1 - (s*f));
	double t = v*(1 - s*(1 - f));
	rgb c;
	switch (hi) {
	case 1:
		c = { int(q*255),int(v * 255),int(p * 255) };
		break;
	case 2:
		c = { int(p * 255),int(v * 255),int(t * 255) };
		break;
	case 3:
		c = { int(p * 255),int(q * 255),int(v * 255) };
		break;
	case 4:
		c = { int(t * 255),int(p * 255),int(v * 255) };
		break;
	case 5:
		c = { int(v * 255),int(p * 255),int(q * 255) };
		break;
	default:
		c = { int(v * 255),int(t * 255),int(p * 255) };
	}
	
	return c;
}

void getdata() {
	bool flag = false;
	ifstream infile(inputf);
	double x, y;
	int count = 0;
	while (infile >> x >> y)
	{
		int i = 0, j = 0;
		if (!flag) {
			isize_x = x;
			isize_y = y;
			X.zeros(x, y);
			Y.zeros(x, y);
			flag = true;
		}
		else {
			i = (count - 1) / isize_y;
			j = (count - 1) % isize_y;
			X(i, j) = y;
			Y(i, j) = x;
		}
		count++;
	}
	
}

void getnoise() {
	ifstream infile2("../noise.txt"); // Found this random noise image online, precomputed its RGB values and stored them in a text file
	double val;
	int count = 0;
	count = 0;
	while (infile2 >> val) {
		int i = (count) / 512;
		int j = (count) % 512;
		NOISE(i, j) = val;
		count++;
	}
}

// == Reading AMIRAMESH files ==============================================================
const char* FindAndJump(const char* buffer, const char* SearchString) {
	const char* FoundLoc = strstr(buffer, SearchString);
	if (FoundLoc) return FoundLoc + strlen(SearchString);
	return buffer;
}

void readAM() {
	const char* FileName = "../p2/Cylinder2D.am";
	FILE* fp = fopen(FileName, "rb");
	if (!fp)
	{
		printf("Could not find %s\n", FileName);
		return;
	}
	printf("Reading %s\n", FileName);
	char buffer[2048];
	fread(buffer, sizeof(char), 2047, fp);
	buffer[2047] = '\0';
	if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
	{
		printf("Not a proper AmiraMesh file.\n");
		fclose(fp);
		return;
	}
	//Find the Lattice definition, i.e., the dimensions of the uniform grid
	int xDim(0), yDim(0), zDim(0);
	sscanf(FindAndJump(buffer, "define Lattice"), "%d %d %d", &xDim, &yDim, &zDim);
	printf("\tGrid Dimensions: %d %d %d\n", xDim, yDim, zDim);
	//Find the BoundingBox
	float xmin(1.0f), ymin(1.0f), zmin(1.0f);
	float xmax(-1.0f), ymax(-1.0f), zmax(-1.0f);
	sscanf(FindAndJump(buffer, "BoundingBox"), "%g %g %g %g %g %g", &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
	printf("\tBoundingBox in x-Direction: [%g ... %g]\n", xmin, xmax);
	printf("\tBoundingBox in y-Direction: [%g ... %g]\n", ymin, ymax);
	printf("\tBoundingBox in z-Direction: [%g ... %g]\n", zmin, zmax);
	//Is it a uniform grid? We need this only for the sanity check below.
	const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);
	printf("\tGridType: %s\n", bIsUniform ? "uniform" : "UNKNOWN");
	//Type of the field: scalar, vector
	int NumComponents(0);
	if (strstr(buffer, "Lattice { float Data }"))
	{
		NumComponents = 1;
	}
	else
	{
		//A field with more than one component, i.e., a vector field
		sscanf(FindAndJump(buffer, "Lattice { float["), "%d", &NumComponents);
	}
	printf("\tNumber of Components: %d\n", NumComponents);
	//Sanity check
	if (xDim <= 0 || yDim <= 0 || zDim <= 0
		|| xmin > xmax || ymin > ymax || zmin > zmax
		|| !bIsUniform || NumComponents <= 0)
	{
		printf("Something went wrong\n");
		fclose(fp);
		return;
	}

	//Find the beginning of the data section
	const long idxStartData = strstr(buffer, "# Data section follows") - buffer;
	if (idxStartData > 0)
	{
		//Set the file pointer to the beginning of "# Data section follows"
		fseek(fp, idxStartData, SEEK_SET);
		//Consume this line, which is "# Data section follows"
		fgets(buffer, 2047, fp);
		//Consume the next line, which is "@1"
		fgets(buffer, 2047, fp);

		//Read the data
		// - how much to read
		const size_t NumToRead = xDim * yDim * zDim * NumComponents;
		x_size = xDim;
		y_size = yDim;
		t_size = zDim;
		mat xx = zeros(xDim, yDim);
		mat yy = zeros(xDim, yDim);
		// - prepare memory; use malloc() if you're using pure C
		float* pData = new float[NumToRead];
		if (pData)
		{
			// - do it
			const size_t ActRead = fread((void*)pData, sizeof(float), NumToRead, fp);
			// - ok?
			if (NumToRead != ActRead)
			{
				printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
				delete[] pData;
				fclose(fp);
				return;
			}

			//Test: Print all data values
			//Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
			int Idx(0);
			int mcounter = 1;
			for (int k = 0; k<zDim; k++)
			{
				for (int j = 0; j<yDim; j++)
				{
					for (int i = 0; i<xDim; i++)
					{
						//Note: Random access to the value (of the first component) of the grid point (i,j,k):
						// pData[((k * yDim + j) * xDim + i) * NumComponents]
						assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);
						for (int c = 0; c<NumComponents; c++)
						{
							if (c == 0) {
								xx(i, j) = double(pData[Idx * NumComponents + c]);
							}
							else {
								yy(i, j) = double(pData[Idx * NumComponents + c]);
							}
							//printf("%g ", pData[Idx * NumComponents + c]);
						}
						//printf("\n");
						Idx++;
					}
				}
				timed_X_raw.push_back(xx);
				timed_Y_raw.push_back(yy);
			}

			delete[] pData;
		}
	}
	printf("\n");
	printf("File was succesfully read and data were loaded to matrices. ==================================\n");
	fclose(fp);

	return;
}
// =========================================================================================
void interpolate(int xs, int ys, mat &X, mat &XX, mat &Y, mat &YY) {
	vec initx_sp = linspace<vec>(0, 511, xs);
	vec inity_sp = linspace<vec>(0, 511, ys);
	vec result_sp = linspace<vec>(0, 511, 512);
	#pragma omp parallel for // Horizontal Linear interpolation in parallel
	for (int u = 0; u < xs; u++) {
		vec x_data = X.row(u).t();
		vec y_data = Y.row(u).t();
		vec x_r, y_r;
		interp1(inity_sp, x_data, result_sp, x_r);
		XX.row(u) = x_r.t();
		interp1(inity_sp, y_data, result_sp, y_r);
		YY.row(u) = y_r.t();
	}
	#pragma omp parallel for // Vertical Linear interpolation in parallel
	for (int u = 0; u < 512; u++) {
		vec x_data = XX.submat(0, 0, xs-1, 511).col(u);
		vec y_data = YY.submat(0, 0, xs-1, 511).col(u);
		vec x_r, y_r;
		interp1(initx_sp, x_data, result_sp, x_r);
		XX.col(u) = x_r;
		interp1(initx_sp, y_data, result_sp, y_r);
		YY.col(u) = y_r;
	}
}

void map_colors(mat &XXX, mat&YYY) {
	mat angle(512, 512);
	mat length(512, 512);
	#pragma omp parallel for
	for (int i = 0; i < 512; i++) {
		#pragma omp parallel for
		for (int j = 0; j < 512; j++) {
			double x = XXX(i, j);
			double y = YYY(i, j);
			angle(i, j) = ((atan2(y, x)+ M_PI) / M_PI) * 180.0;
			length(i, j) = sqrt(pow(x, 2) + pow(y, 2));
		}
	}
	double max_l = length.max();
	double min_l = length.min();
	length = (length - min_l);
	double temp = 1 / (max_l - min_l);
	mat dividee(512, 512);
	dividee.fill(temp);
	length = length % dividee;
	int track = 0;
	board = new unsigned char[512 * 512 * 3];
	for (int i = 0; i < 512; i++) {
		for (int j = 0; j < 512; j++) {
			rgb r = hsv2rgb(angle(i, j), length(i, j));
			board[track] = r.r;
			board[track + 1] = r.g;
			board[track + 2] = r.b;
			track += 3;
		}
	}
}

vector <V> compute_integral_curve(mat &XXX, mat &YYY,int i, int j, int L, int ds) {
	vector <V> C;
	V vect = { XXX(i,j),YYY(i,j) };
	V P = { i,j };
	C.push_back(P);
	double x = i, y = j;
	for (int s = 0; s < L; s++) {
		x = x + ds*vect.x;
		y = y + ds*vect.y;
		V temp = { int(x),int(y) };
		if (x < 512 && y < 512 && x >= 0 && y >= 0) {
			vect.x = XXX(int(x), int(y));
			vect.y = YYY(int(x), int(y));
			C.push_back(temp);
		}
	}
	x = i, y = j;
	for (int s = 0; s > -L; s--) {
		x = x - ds*vect.x;
		y = y - ds*vect.y;
		V temp = { int(x),int(y) };
		if (x < 512 && y < 512 && x >= 0 && y >= 0) {
			vect.x = XXX(int(x), int(y));
			vect.y = YYY(int(x), int(y));
			C.push_back(temp);
		}
	}
	return C;
}

int compute_convolution(vector <V> C, int L)  {
	double sum = 0;
	for(int i = 0; i < C.size(); i++) {
		sum = sum + NOISE(C[i].x, C[i].y);
	}
	sum = sum / (4 * L + 1); // Normalization : 4 because gives smoother result
	sum *= 255/ (4 * L + 1);
	return int(sum);
}

mat LIC(mat& XXX, mat &YYY, int L, int ds) {
	clock_t tStart = clock();
	mat LIC_RESULT = zeros(512, 512);
	#pragma omp parallel for
	for (int i = 0; i < 512; i++) {
		#pragma omp parallel for
		for (int j = 0; j < 512; j++) {
			vector <V> c = compute_integral_curve(XXX,YYY,i, j,L,ds);
			int sum = compute_convolution(c,L);
			LIC_RESULT(i, j) = sum;
		}
	}
	printf("Time taken to compute LIC: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	return LIC_RESULT;
}

// Boards the computed LIC data of an image on an array for displaying
void LIC_board(mat &data) {
	int count = 0;
	board2 = new unsigned char[512 * 512 * 3];
	for (int i = 0; i < 512; i++) {
		for (int j = 0; j < 512; j++) {
			board2[count] = data(i, j);
			board2[count + 1] = data(i, j);
			board2[count + 2] = data(i, j);
			count += 3;
		}
	}
}

void streak_board(vector <V> p) {
	board2 = new unsigned char[512 * 512 * 3];
	fill_n(board2, 512 * 512 * 3, 255);
	for (int u = 0; u < p.size(); u++) {
		int i = p[u].x;
		int j = p[u].y;
		if (i > 511 || i < 0 || j > 511 || j < 0)
			continue;
		board2[512 * 3 * j + i * 3] = 255;
		board2[512 * 3 * j + i * 3 + 1] = 0;
		board2[512 * 3 * j + i * 3 + 2] = 0;
	}
}

// Generates (m x m) seed points in grid fashion.  ( m= 512/n )
vector <V> generate_seeds(int n) {
	vector <V> result;
	int m = 512 / n;
	for (int k = 0; k < 512; k += m) {
		for (int e = 0; e < 512; e += m) {
			V r = { k,e };
			result.push_back(r);
		}
	}
	return result;
}

// Generates streamlines, given a vector field
vector <V> streamline(mat &XXX, mat&YYY) {
	int n = 16; // to generate a 32x32 seed points in grid fashion
	vector <V> seeds = generate_seeds(n);
	vector <V> map;
	for (int u = 0; u < seeds.size(); u++) {
		int i = seeds[u].x;
		int j = seeds[u].y;
		V k1, k2, k3, k4;
		map.push_back(seeds[u]);
		double new_i = i;
		double new_j = j;
		double old_i = -1;
		double old_j = -1;
		double dt = 1;
		int count = 3000;
		// The following loop, grows the streamline in positve direction of vector at the seed point
		while (true && count > 0) {
			double x = XXX(new_i, new_j);
			double y = YYY(new_i, new_j);
			k1.x = dt*x;
			k1.y = dt*y;
			k2.x = dt*(XXX(new_i + k1.x / 2, new_j + k1.y / 2));
			k2.y = dt*(YYY(new_i + k1.x / 2, new_j + k1.y / 2));
			k3.x = dt*(XXX(new_i + k2.x / 2, new_j + k2.y / 2));
			k3.y = dt*(YYY(new_i + k2.x / 2, new_j + k2.y / 2));
			k4.x = dt*(XXX(new_i + k3.x, new_j + k3.y));
			k4.y = dt*(YYY(new_i + k3.x, new_j + k3.y));
			new_i = new_i + k1.x / 6 + k2.x / 3 + k3.x / 3 + k4.x / 6;
			new_j = new_j + k1.y / 6 + k2.y / 3 + k3.y / 3 + k4.y / 6;
			if (new_i > 511 || new_i < 0 || new_j > 511 || new_j < 0) {
				break;
			}
			V temp = { new_i,new_j };
			if (!(int(new_i) == int(old_i) && int(new_j) == int(old_j))) {
				map.push_back(temp);
				old_i = new_i;
				old_j = new_j;
				count--;
			}
		}
		count = 3000;
		new_i = i;
		new_j = j;
		// The following loop, grows the streamline in negative direction of vector at the seed point
		while (true && count > 0) {
			double x = XXX(new_i, new_j);
			double y = YYY(new_i, new_j);
				k1.x = dt*-x;
				k1.y = dt*-y;
				k2.x = dt*(-XXX(new_i + k1.x / 2, new_j + k1.y / 2));
				k2.y = dt*(-YYY(new_i + k1.x / 2, new_j + k1.y / 2));
				k3.x = dt*(-XXX(new_i + k2.x / 2, new_j + k2.y / 2));
				k3.y = dt*(-YYY(new_i + k2.x / 2, new_j + k2.y / 2));
				k4.x = dt*(-XXX(new_i + k3.x, new_j + k3.y));
				k4.y = dt*(-YYY(new_i + k3.x, new_j + k3.y));
				new_i = new_i + k1.x / 6 + k2.x / 3 + k3.x / 3 + k4.x / 6;
				new_j = new_j + k1.y / 6 + k2.y / 3 + k3.y / 3 + k4.y / 6;
			if (new_i > 511 || new_i < 0 || new_j > 511 || new_j < 0) {
				break;
			}
			V temp = { new_i,new_j };
			if (!(int(new_i) == int(old_i) && int(new_j) == int(old_j))) {
				map.push_back(temp);
				old_i = new_i;
				old_j = new_j;
				count--;
			}
		}
	}
	return map;
}

vector <V> trace_points(vector<V> &points,mat &XXX,mat &YYY){
	vector <V> result;
	for (int i = 0; i < points.size(); i++) {
		double xx = points[i].x;
		double yy = points[i].y;
		if (points[i].x > 511 || points[i].x < 0 || points[i].y > 511 || points[i].y < 0) {
			continue;
		}
		points[i].x = xx + int (XXX(xx, yy));
		points[i].y = yy + int (YYY(xx, yy));
		if (!(xx == points[i].x && yy == points[i].y)) {
			result.push_back({ xx,yy });
			result.push_back(points[i]);
		}
	}
	return result;
}

// Boards the computed streamline data of an image on an array for displaying
void board_streamline(vector<V>& map) {
	board2 = new unsigned char[512 * 512 * 3];
	fill_n(board2, 512 * 512 * 3, 255);
	for (int u = 0; u < map.size(); u++) {
		int i = map[u].x;
		int j = map[u].y;
		if (i > 511 || i < 0 || j >511 || j < 0) {
			continue;
		}
		board2[512 * 3 * i + j * 3] = 0;
		board2[512 * 3 * i + j * 3 + 1] = 0;
		board2[512 * 3 * i + j * 3 + 2] = 0;
	}
}

// Writes precomputed LIC data
void write_anim(string flag) {
	if (flag == "lic") {
		ofstream out("../lics.txt");
		readAM();
		mat temp, tx, ty;
		for (int u = 0; u < 30; u++) {
			temp = zeros(512, 512);
			Xi.push_back(temp);
			Yi.push_back(temp);
			interpolate(x_size, y_size, timed_X_raw[u], Xi[u], timed_Y_raw[u], Yi[u]);
			tx = Xi[u];
			ty = Yi[u];
			temp = LIC(tx, ty, 25, 40).t();
			//lics.push_back(temp);
			for (int i = 0; i < 512; i++) {
				for (int j = 0; j < 512; j++) {
					out << temp(i, j) << " ";
				}
			}
			cout << u + 1 << ": LIC image pushed " << endl;
		}
		out.close();
	}
	else if(flag == "stream") {
		ofstream out("../streamline.txt");
		readAM();
		mat temp, tx, ty;
		for (int u = 0; u < 30; u++) {
			temp = zeros(512, 512);
			vector <V> r;
			Xi.push_back(temp);
			Yi.push_back(temp);
			interpolate(x_size, y_size, timed_X_raw[u], Xi[u], timed_Y_raw[u], Yi[u]);
			tx = Xi[u];
			ty = Yi[u];
			r = streamline(tx, ty);
			streamlines.push_back(r);
			for (int u = 0; u < r.size(); u++) {
				int i = r[u].x;
				int j = r[u].y;
				out << i << " " << j << " ";
			}
			out << 999 << " ";
			cout << u + 1 << ": Streamline image pushed " << endl;
		}
		out.close();
	}
	else {
		ofstream out("../streaks.txt");
		readAM();
		mat temp, tx, ty;
		vector <V> pp;
		pp = generate_seeds(64);
		for (int u = 0; u < 5; u++) {
			temp = zeros(512, 512);
			vector <V> r;
			Xi.push_back(temp);
			Yi.push_back(temp);
			interpolate(x_size, y_size, timed_X_raw[u], Xi[u], timed_Y_raw[u], Yi[u]);
			tx = Xi[u];
			ty = Yi[u];
			streaks.push_back(pp);
			pp = trace_points(pp, tx, ty);
			for (int i = 0; i < pp.size(); i++) {
				if (pp[i].x < 512 && pp[i].x >= 0 && pp[i].y < 512 && pp[i].y >= 0)
					out << pp[i].x << " " << pp[i].y << " ";
			}
			out << 999 << " ";
			cout << u << ": frame pushed" << endl;
		}
	}
}

// Reads precomputed LIC data
void read_anim() {
	ifstream infile2("../lics.txt");
	int count;
	mat temp;
	count = 0;
	int mcount = 0;
	int i, j;
	int val;
	while (mcount < 30) {
		count = 0;
		temp = zeros(512, 512);
		while (infile2 >> val) {
			i = count / 512;
			j = count % 512;
			temp(i, j) = val;
			count++;
			if (count == 262144)
				break;
		}
		mcount++;
		lics.push_back(temp);
	}
}

void renderScene(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(512, 512, GL_RGB, GL_UNSIGNED_BYTE, board);
	glutSwapBuffers();
}

void renderScene2(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(512, 512, GL_RGB, GL_UNSIGNED_BYTE, board2);
	glutSwapBuffers();
}

void renderScene3(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	LIC_board(lics[frame]);
	frame++;
	if (frame == 30)
		frame = 0;
	glDrawPixels(512, 512, GL_RGB, GL_UNSIGNED_BYTE, board2);
	delete[] board2;
	glutSwapBuffers();
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(512, 512);
	glutCreateWindow("ECE522 - Assignment 2");
	frame = 0;
	int ch1= 0, ch2=0;
	cout << "================================================================================" << endl;
	cout << "SEYEDMOEIN MIRHOSSEINI    20165447    DATA VISUALIZATION : ASSIGNMENT2" << endl;
	cout << "================================================================================" << endl;
	cout << "Please select an input file. (Enter a number)" << endl;
	cout << "1.bnoise    2.bruno3    3.dipole    4.Cylinder (for part II only)" << endl;
	cin >> ch1;
	cout << "================================================================================" << endl;
	cout << "Please select a visualization method" << endl << endl;
	cout << "Part I ==============================================" << endl;
	cout << "1.Color Map    2.LIC    3.Streamline" << endl << endl;;
	cout << "Part II (Animated Visualization) ====================" << endl;
	cout << "4.LIC    5.Streaklines" << endl;
	cin >> ch2;
	cout << endl << "Please wait ... (It could take a while)" << endl;
	switch (ch1) {
	case 1:
		inputf = "../bnoise.vec";
		getdata();
		break;
	case 2:
		inputf = "../bruno3.vec";
		getdata();
		break;
	case 3:
		inputf = "../dipole.vec";
		getdata();
		break;
	case 4:
		//readAM();
		break;
	}
	switch (ch2) {
	case 1:
		interpolate(isize_x, isize_y, X, XX, Y, YY);
		map_colors(XX, YY);
		glutDisplayFunc(renderScene);
		break;
	case 2:
		getnoise();
		interpolate(isize_x, isize_y, X, XX, Y, YY);
		LIC_board(LIC(XX, YY, 25, 13));
		glutDisplayFunc(renderScene2);
		break;
	case 3:
		interpolate(isize_x, isize_y, X, XX, Y, YY);
		board_streamline(streamline(XX, YY));
		glutDisplayFunc(renderScene2);
		break;
	case 4:
		read_anim();
		glutDisplayFunc(renderScene3);
		glutIdleFunc(renderScene3);
		break;
	case 5:
		cout << endl << "This method is not implemented" << endl;
		break;
	}
	glutMainLoop();

	return 1;
}