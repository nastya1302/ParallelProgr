#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <string>
#include <random>
#include <mpi.h>

using namespace std;

const string BASE_DIR = "../";
const string MATRIX_A_DIR = BASE_DIR + "matrix_a/";
const string MATRIX_B_DIR = BASE_DIR + "matrix_b/";
const string RESULT_DIR = BASE_DIR + "results/";

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

void saveResult(int size, const vector<vector<int>>& matrix) {
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
    if (!log) throw runtime_error("Cannot open log file");

    log << "Size: " << size << "x" << size
        << " | Avg time: " << average_time << " sec"
        << " | Accuracy: " << accuracy << "%"
        << endl;
}

vector<vector<int>> generateSquareMatrix(int size) {
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

vector<vector<int>> multiplyMatrixMPI(const vector<vector<int>>& matrixA, const vector<vector<int>>& matrixB, int size, int rank, int num_procs) {
    int error_code = 0;
    if (rank == 0) {
        if (matrixA.empty() || matrixB.empty()) error_code = 1;
        else if (matrixA.size() != matrixB.size()) error_code = 2;
    }
    MPI_Bcast(&error_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != 0) return {};

    vector<vector<int>> localB(size);
    if (rank == 0) {
        localB = matrixB;
    }
    for (int i = 0; i < size; ++i) {
        localB[i].resize(size);
        MPI_Bcast(localB[i].data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    vector<vector<int>> localA;
    int rows_per_proc = size / num_procs;
    int remainder = size % num_procs;
    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    if (rank == 0) {
        vector<int> counts(num_procs), displs(num_procs);
        for (int i = 0; i < num_procs; ++i) {
            counts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * size;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }

        vector<int> send_buffer;
        send_buffer.reserve(size * size);
        for (const auto& row : matrixA) {
            send_buffer.insert(send_buffer.end(), row.begin(), row.end());
        }

        vector<int> recv_buffer(counts[0]);
        MPI_Scatterv(send_buffer.data(), counts.data(), displs.data(), MPI_INT,
            recv_buffer.data(), counts[0], MPI_INT, 0, MPI_COMM_WORLD);

        localA.resize(my_rows, vector<int>(size));
        for (int i = 0; i < my_rows; ++i) {
            copy(recv_buffer.begin() + i * size, recv_buffer.begin() + (i + 1) * size, localA[i].begin());
        }
    }
    else {
        vector<int> recv_buffer(my_rows * size);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
            recv_buffer.data(), my_rows * size, MPI_INT, 0, MPI_COMM_WORLD);

        localA.resize(my_rows, vector<int>(size));
        for (int i = 0; i < my_rows; ++i) {
            copy(recv_buffer.begin() + i * size, recv_buffer.begin() + (i + 1) * size, localA[i].begin());
        }
    }

    vector<vector<int>> local_result(my_rows, vector<int>(size, 0));
    for (int i = 0; i < my_rows; ++i) {
        for (int k = 0; k < size; ++k) {
            for (int j = 0; j < size; ++j) {
                local_result[i][j] += localA[i][k] * localB[k][j];
            }
        }
    }

    vector<vector<int>> result;
    if (rank == 0) {
        result.resize(size, vector<int>(size));
    }

    vector<int> send_buffer;
    send_buffer.reserve(local_result.size() * size);
    for (const auto& row : local_result) {
        send_buffer.insert(send_buffer.end(), row.begin(), row.end());
    }

    if (rank == 0) {
        vector<int> counts(num_procs), displs(num_procs);
        for (int i = 0; i < num_procs; ++i) {
            counts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * size;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }

        vector<int> recv_buffer(size * size);
        MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_INT,
            recv_buffer.data(), counts.data(), displs.data(), MPI_INT,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                copy(recv_buffer.begin() + i * size, recv_buffer.begin() + (i + 1) * size, result[i].begin());
            }
        }
    }
    else {
        MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_INT,
            nullptr, nullptr, nullptr, MPI_INT,
            0, MPI_COMM_WORLD);
    }

    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int NUM_ITERATIONS = 10;
    const int MIN_SIZE = 100;
    const int MAX_SIZE = 1500;
    const int STEP_SIZE = 100;

    if (rank == 0) {
        setlocale(LC_ALL, "ru-ru");
        srand(time(nullptr));
    }

    for (int size = MIN_SIZE; size <= MAX_SIZE; size += STEP_SIZE) {
        vector<vector<int>> matrixA, matrixB, result;
        if (rank == 0) {
            matrixA = generateSquareMatrix(size);
            matrixB = generateSquareMatrix(size);
            saveMatrix(MATRIX_A_DIR + "matrix_" + to_string(size) + ".txt", matrixA);
            saveMatrix(MATRIX_B_DIR + "matrix_" + to_string(size) + ".txt", matrixB);
        }

        double total_time = 0.0;
        int correct_count = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();

            result = multiplyMatrixMPI(matrixA, matrixB, size, rank, num_procs);

            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = MPI_Wtime();

            if (rank == 0) {
                double iteration_time = end_time - start_time;
                total_time += iteration_time;

                saveMatrix(RESULT_DIR + "result_" + to_string(size) + ".txt", result);
                bool correct = verifyWithPython(size);
                if (correct) correct_count++;

                cout << "Size: " << size << "x" << size
                    << " | Iter: " << iter + 1
                    << " | Correct: " << (correct ? "Yes" : "No")
                    << endl;
            }
        }

        if (rank == 0) {
            double average_time = total_time / NUM_ITERATIONS;
            double accuracy = (double)correct_count / NUM_ITERATIONS * 100.0;

            ofstream log(BASE_DIR + "result.txt", ios::app);
            log << "Size: " << size << "x" << size
                << " | Avg time: " << average_time << " sec"
                << " | Accuracy: " << accuracy << "%"
                << endl;
        }
    }

    if (rank == 0) {
        system("pause");
    }

    MPI_Finalize();
    return 0;
}