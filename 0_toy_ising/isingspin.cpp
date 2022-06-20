#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <time.h>
using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis(0, 1);

void initialize(vector<double>& v, int size) //initial -random- state
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if ((i + j) % 2 == 0) v[size * i + j] = 1;
			else v[size * i + j] = -1;
		}
	}
	/*for (int i = 0; i < size * size; i++) {
		v[i] = dis(gen) < 0.5 ? 1 : -1;
	}*/
}
void neighbor(vector < vector <double> >& na, int size)
{
	int sizes = size * size;
	for (int i = 0; i < size * size; i++) {
		na[i][0] = (i + size * (size - 1)) % sizes;
		na[i][1] = (i + size) % sizes;
		na[i][2] = (i - 1 + size) % size + (i / size) * size;
		na[i][3] = (i + 1) % size + (i / size) * size;
	}
}
double Magnet(vector<double>& v, int size)
{
	double m = 0;
	for (vector<int>::size_type i = 0; i < v.size(); i++) {
		m = m + v.at(i);
	}
	m = abs(m) / (v.size()); //absolute value of average spin
	return m;
}
void Cluster_1step(vector<double>& v, int size, double padd, vector < vector <double> > na)
{
	int i = size * size * dis(gen);
	double oldspin = v[i];
	double newspin = -v[i];
	vector<int> stack(size*size, 0);
	stack[0] = i; v[i] = newspin;
	int sp = 0, sh = 0;
	while (1) {
		if (v[na[i][0]] == oldspin && dis(gen) < padd) { sh++; stack.at(sh) = na[i][0]; v[na[i][0]] = newspin; }
		if (v[na[i][1]] == oldspin && dis(gen) < padd) { sh++; stack.at(sh) = na[i][1]; v[na[i][1]] = newspin; }
		if (v[na[i][2]] == oldspin && dis(gen) < padd) { sh++; stack.at(sh) = na[i][2]; v[na[i][2]] = newspin; }
		if (v[na[i][3]] == oldspin && dis(gen) < padd) { sh++; stack.at(sh) = na[i][3]; v[na[i][3]] = newspin; }
		sp++;
		if (sp > sh) break;
		i = stack.at(sp);
	}
}
void MC_1cycle(int size, double T, double& mag, vector < vector <double> > na, vector<double>& array)
{
	int step1 = 2500, step2 = 5000;
	int scale=1;
    double Tstart = 2.3, clsizef = 2.86;
	double slope = (double(size)*size/clsizef)/(5-Tstart);
	if (T>Tstart) { scale = slope * (T - Tstart); if (scale == 0) scale = 1; }
	int trash_step = scale*(sqrt(size));

	double padd = 1 - exp(-2 / T);
	for (int k = 0; k < step1; k++) { Cluster_1step(array, size, padd, na); }

	vector<double> magnet(step2, 0);
	for (int k = 0; k < step2; k++) {
		for (int h = 0; h < trash_step; h++) {
			Cluster_1step(array, size, padd, na);
		}
		magnet.at(k) = Magnet(array, size);
	}

	double Mag = 0;
	for (vector<int>::size_type i = 0; i < magnet.size(); i++) {
		Mag = Mag + magnet.at(i);
	}
	mag = Mag / step2;
}
int main()
{
	int size = 32;
	double Mag = 0;

	vector < vector <double> > near(size * size, vector<double>(4, 0));
	vector<double> array(size * size, 0);
    neighbor(near, size);
	initialize(array, size);

	clock_t start = clock();

	

	for (int k = 28; k < 30; k++) { // T = 0.02~5.82
		string knum = to_string(k);
		ofstream File;
		File.open("spin32_"+ knum +".txt");
		cout << "File open, temp=" << 0.2*k+0.02 << endl;
		File << "temp m spin " << endl;
		for (int h = 0; h < 200; h++) {
			MC_1cycle(size, 0.2 * k + 0.02, Mag, near, array);
			File << 0.2 * k + 0.02 << " " << Mag << " ";
            for (int i = 0; i < size*size; i++){
                File << array.at(i) << " ";
            }
            File << endl;
		}
		cout << 0.2 * k + 0.02 << "end" << endl;
		File.close();
	}
	


	cout << endl << "total time: " << (double(clock()) - double(start)) / CLOCKS_PER_SEC << " sec" << endl;
	return 0;
}
