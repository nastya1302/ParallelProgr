#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <string>
#include <random>
#include <omp.h>

using namespace std;
const string BASE_DIR = "../";
const string MATRIX_A_DIR = BASE_DIR + "matrix_a/";
const string MATRIX_B_DIR = BASE_DIR + "matrix_b/";
const string RESULT_DIR = BASE_DIR + "results/";

vector<vector<int>> generateMatrix(int size) {
    vector<vector<int>> matrix(size, vector<int>(size));
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<> distrib(-100, 100);
    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = distrib(generator);;
        }
    }
    return matrix;
}

void saveMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream out(filename);
    if (!out) throw runtime_error("Cannot create file: " + filename);

    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << endl;
    }
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& a, const vector<vector<int>>& b) {
    const int n = a.size();
    vector<vector<int>> result(n, vector<int>(n, 0));
    #pragma omp parallel for shared(a, b, result) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            int a_ik = a[i][k];
            for (int j = 0; j < n; ++j) {
                result[i][j] += a_ik * b[k][j];
            }
        }
    }
    return result;
}

void saveResult(int size, int iter, const vector<vector<int>>& matrix) {
    string filename = RESULT_DIR + "result_" + to_string(size) + ".txt";
    ofstream out(filename);
    if (!out) throw runtime_error("Cannot create result file");

    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << endl;
    }
}

bool verifyWithPython(int size, int iter) {
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
    if (!log) throw runtime_error("Cannot open log file");

    log << "Size: " << size << "x" << size
        << " | Avg time: " << average_time << " sec"
        << " | Accuracy: " << accuracy << "%"
        << endl;
}

int main() {
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
                auto result = multiplyMatrices(matrixA, matrixB);
                clock_t end = clock();

                saveResult(size, iter, result);
                double duration = double(end - start) / CLOCKS_PER_SEC;
                total_time += duration;

                bool correct = verifyWithPython(size, iter);
                if (correct) correct_count++;

                cout << "Размер: " << size << "x" << size
                    << " | Итерация: " << iter + 1
                    << " | Корректно: " << (correct ? "Да" : "Нет")
                    << endl;
            }
            catch (const exception& e) {
                cerr << "Ошибка: " << e.what() << endl;
            }
        }

        double accuracy = (double)correct_count / NUM_ITERATIONS * 100.0;
        logResults(size, total_time / NUM_ITERATIONS, accuracy);
    }

    system("pause");
    return 0;
}