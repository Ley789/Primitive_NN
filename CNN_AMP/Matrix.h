#pragma once
#include <amp.h>

//TODO create optimized version for convolution
//By changing the order (e.g. store the columns of the 2 matrix)
//Fisrt version
template<typename T>
class Matrix
{
	T *mat;
	int rows;
	int col;
public:
	Matrix(int r, int c) {
		rows = r;
		col = c;
		mat = new T [rows * col];
	};
	~Matrix() {
		delete[] mat;
	};
	//Overriden operators
	inline T* operator[](const unsigned int i);
	inline bool operator==(const Matrix<T> &rhs);
	void MatrixMultiplicationSeq(Matrix<T> &aMat, Matrix<T> &bMat);
	void MatrixMultiplication(Matrix<T> &aMat, Matrix<T> &bMat);
	void MatrixVectorMultiplication(Matrix<T> &aMat, Matrix<T> &bMat);

	//Getter
	inline int GetRows() {
		return rows;
	}
	inline int GetColumns() {
		return col;
	}
	inline T* GetData() {
		return mat;
	}
};

#pragma region Overriden operatiors

template <typename T>
inline T* Matrix<T>::operator[](const unsigned int i)
{
	return &mat[col * i];
}

template <typename T>
inline bool Matrix<T>::operator==(const Matrix<T> &rhs){
	if (rows != rhs.rows) return false;
	if (col != rhs.col) return false;
	int outerIndex;
	for (int i = 0; i < rows; i++) {
		outerIndex = i * col;
		for (int j = 0; j < col; j++) {
			if (mat[outerIndex + j] != rhs.mat[outerIndex + j])return false;
		}
	}
	return true;
}

#pragma endregion

//Parallel version
template <typename T>
void Matrix<T>::MatrixMultiplication(Matrix<T> &aMat, Matrix<T> &bMat) {
	//TODO dimension check

	concurrency::array_view<T, 2> a(aMat.GetRows(), aMat.GetColumns(), aMat.mat);
	concurrency::array_view<T, 2> b(bMat.GetRows(), bMat.GetColumns(), bMat.mat);
	concurrency::array_view<T, 2> r(rows, col, mat);

	int inner = aMat.col;
	concurrency::parallel_for_each(
		r.extent,
		[=](concurrency::index<2> idx) restrict(amp) {
		r[idx] = 0;
		int row = idx[0];
		int col = idx[1];
		for (int i = 0; i < inner; i++) {
			r[idx] += a(row, i) * b(i, col);
		}
	}
	);
}

/*
	This method assumes that bMat has 1 row and uses this rows as column vector to perform
	aMat * bVec
*/
template <typename T>
void Matrix<T>::MatrixVectorMultiplication(Matrix<T> &aMat, Matrix<T> &bVec) {
	//TODO dimension check

	concurrency::array_view<T, 2> a(aMat.rows, aMat.col, aMat.mat);
	concurrency::array_view<T, 1> b(bVec.col, bVec.mat);
	concurrency::array_view<T, 1> r(rows, col, mat);

	int inner = aMat.col;
	concurrency::parallel_for_each(
		r.extent,
		[=](concurrency::index<1> idx) restrict(amp) {
		r[idx] = 0;
		int row = idx[0];
		for (int i = 0; i < inner; i++) {
			r[idx] += a(row, i) * b(i);
		}
	}
	);
}

template<typename T, int size>
void MaxPooling(Matrix<T> &res, Matrix<T> &input){
	//TODO check that this mat is smaller then input mat and
	//both can be divided by size
	concurrency::array_view<T, 2> a(input.GetRows(), input.GetColumns(), input.GetData());
	concurrency::array_view<T, 2> b(res.GetRows(), res.GetColumns(), res.GetData());
	
	concurrency::parallel_for_each(
		a.extent.tile<size, size>(),
		[=](concurrency::tiled_index<size, size> t_idx) restrict(amp) {

		tile_static T num[size][size];
		num[t_idx.local[0]][t_idx.local[1]] = a[t_idx.global[0]][t_idx.global[1]];
		t_idx.barrier.wait();
		T max = num[0][0];
		for (int j = 1; j < size; j++) {
			if (max < num[0][j]) {
				max = num[0][j];
			}
		}
		for (int i = 1; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (max < num[i][j]) {
					max = num[i][j];
				}
			}
		}
		if (t_idx.local[0] == 0 && t_idx.local[1] == 0) {
			b[t_idx.tile[0]][t_idx.tile[1]] = max;
		}
	});
}

/* Debuging functions */

template <typename T>
void Matrix<T>::MatrixMultiplicationSeq(Matrix<T> &aMat, Matrix<T> &bMat){
	//TODO Check dimension

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < col; j++) {
			mat[i * col + j] = 0;
			for (int k = 0; k < aMat.GetColumns(); k++) {
				mat[i * col + j] += aMat[i][k] * bMat[k][j];
			}
		}
	}
}
