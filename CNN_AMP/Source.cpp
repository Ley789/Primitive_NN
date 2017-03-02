#include <iostream>
#include "Matrix.h"
#include "Model.h"

using namespace std;

constexpr int M = 3;
constexpr int N = 2;


int main(void) {
	std::vector<float> input(0);
	input.push_back(2.0);
	input.push_back(2.5);
	input.push_back(3.0);
	input.push_back(1.0);
	input.push_back(1.5);
	Model m(5);
	m.AddFullNetworkLayer(1);
	m.Finish();
	auto res = m.GetHyp(input);
	for (unsigned int i = 0; i < res.size(); i++) {
		cout << res[i] << endl;
	}
	std::string t;
	getline(std::cin, t);
}


/*

int aMatrix[2][3] = { {1, 2, 3}, { 4, 5, 6 } };
int bMatrix[3][3] = { {1, 2, 3 }, { 4, 5, 6 },{ 1, 2, 3 } };
Matrix<int> res1(2, 3);
Matrix<int> res2(2, 3);
Matrix<int> aMat(2, 3);
aMat[0][0] = 1;
aMat[0][1] = 2;
aMat[0][2] = 3;
aMat[1][0] = 4;
aMat[1][1] = 5;
aMat[1][2] = 6;
Matrix<int> bMat(3, 3);
bMat[0][0] = 1;
bMat[0][1] = 2;
bMat[0][2] = 3;
bMat[1][0] = 4;
bMat[1][1] = 5;
bMat[1][2] = 6;
bMat[2][0] = 1;
bMat[2][1] = 2;
bMat[2][2] = 3;
Matrix<int> cMat(4, 4);
cMat[0][0] = 1;
cMat[0][1] = 2;
cMat[0][2] = 3;
cMat[0][3] = 3;
cMat[1][0] = 4;
cMat[1][1] = 5;
cMat[1][2] = 6;
cMat[1][3] = 3;
cMat[2][0] = 1;
cMat[2][1] = 2;
cMat[2][2] = 3;
cMat[2][3] = 10;
cMat[3][0] = 4;
cMat[3][1] = 5;
cMat[3][2] = 6;
cMat[3][3] = 3;
Matrix<int> pool(2, 2);


res1.MatrixMultiplication(aMat, bMat);
res2.MatrixMultiplication(aMat, bMat);
MaxPooling<int, 2>(pool, cMat);
for (int i = 0; i < pool.GetRows(); i++) {
for (int j = 0; j < pool.GetColumns(); j++) {
cout << pool[i][j] << " ";
}
cout << endl;
}
cout << (res1 == res2) << endl;
std::string t;
getline(std::cin, t);


*/