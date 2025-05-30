#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

const string BASE_DIR = "../";
const string MATRIX_A_DIR = BASE_DIR + "matrix_a/";
const string MATRIX_B_DIR = BASE_DIR + "matrix_b/";
const string RESULT_DIR = BASE_DIR + "results/";

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << "Error code: " << cudaGetErrorName(err) << " - "
            << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(err) checkCudaError(err, __FILE__, __LINE__)

__global__ void matrixMultiplyKernel(const int* a, const int* b, int* result, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += a[row * size + k] * b[k * size + col];
        }
        result[row * size + col] = sum;
    }
}

vector<vector<int>> generateMatrix(int size) {
    vector<vector<int>> matrix(size, vector<int>(size));
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<> distrib(-100, 100);

    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = distrib(generator);
        }
    }
    return matrix;
}

void saveMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Cannot create file: " + filename);
    }

    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << "\n";
    }
}

vector<vector<int>> multiplyMatricesCUDA(const vector<vector<int>>& a, const vector<vector<int>>& b) {
    const int size = a.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

    if (a[0].size() != size || b.size() != size || b[0].size() != size) {
        throw runtime_error("Matrix dimensions mismatch");
    }

    vector<int> flatA(size * size);
    vector<int> flatB(size * size);
    vector<int> flatResult(size * size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flatA[i * size + j] = a[i][j];
            flatB[i * size + j] = b[i][j];
        }
    }

    int* d_a, * d_b, * d_result;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size * size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size * size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_result, size * size * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_a, flatA.data(), size * size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, flatB.data(), size * size * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32); 
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x,
        (size + blockSize.y - 1) / blockSize.y);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(flatResult.data(), d_result, size * size * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_result));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i][j] = flatResult[i * size + j];
        }
    }

    return result;
}

void saveResult(int size, const vector<vector<int>>& matrix) {
    string filename = RESULT_DIR + "result_" + to_string(size) + ".txt";
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Cannot create result file: " + filename);
    }

        for (const auto& row : matrix) {
            for (int val : row) {
                out << val << " ";
            }
            out << "\n";
        }
}

bool verifyWithPython(int size) {
    string result_file = RESULT_DIR + "result_" + to_string(size) + ".txt";
    string command = "python ../verify.py " +
        MATRIX_A_DIR + "matrix_" + to_string(size) + ".txt " +
        MATRIX_B_DIR + "matrix_" + to_string(size) + ".txt " +
        result_file;

    FILE* pipe = _popen(command.c_str(), "r");
    if (!pipe) return false;

    char buffer[128];
    string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    _pclose(pipe);

    return result.find("True") != string::npos;
}

void logResults(int size, double average_time, double accuracy) {
    ofstream log(BASE_DIR + "result.txt", ios::app);
    if (!log) {
        throw runtime_error("Cannot open log file");
    }

    log << "Size: " << size << "x" << size
        << " | Avg time: " << average_time << " sec"
        << " | Accuracy: " << accuracy << "%"
        << endl;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "Error: No CUDA-capable devices found!" << endl;
        return 1;
    }
    cudaSetDevice(0);

    setlocale(LC_ALL, "ru-ru");
    srand(time(nullptr));

    const int NUM_ITERATIONS = 10;
    const int MIN_SIZE = 100;
    const int MAX_SIZE = 1500;
    const int STEP_SIZE = 100;

    for (int size = MIN_SIZE; size <= MAX_SIZE; size += STEP_SIZE) {
        double total_time = 0.0;
        int correct_count = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            try {
                auto matrixA = generateMatrix(size);
                auto matrixB = generateMatrix(size);

                saveMatrix(MATRIX_A_DIR + "matrix_" + to_string(size) + ".txt", matrixA);
                saveMatrix(MATRIX_B_DIR + "matrix_" + to_string(size) + ".txt", matrixB);

                clock_t start = clock();
                auto result = multiplyMatricesCUDA(matrixA, matrixB);
                clock_t end = clock();

                saveResult(size, result);
                double duration = double(end - start) / CLOCKS_PER_SEC;
                total_time += duration;

                bool correct = verifyWithPython(size);
                if (correct) correct_count++;

                cout << "Size: " << size << "x" << size
                    << " | Iteration: " << iter + 1
                    << " | Time: " << duration << " sec"
                    << " | Correct: " << (correct ? "Yes" : "No")
                    << endl;
            }
            catch (const exception& e) {
                cerr << "Error: " << e.what() << endl;
            }
        }
        double average_time = total_time / NUM_ITERATIONS;
        double accuracy = (double)correct_count / NUM_ITERATIONS * 100.0;
        logResults(size, average_time, accuracy);
    }

    return 0;
}