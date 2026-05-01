#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

using namespace std;

// --- Constants -----------------------------------------------
#define MAX_N 2000          // Safety cap so we don't exhaust memory
#define FILL_RANGE 10       // Random values in [1, FILL_RANGE]

// --- Matrix helpers (flat 1-D arrays for cache friendliness) --
typedef double* Matrix;

Matrix allocMatrix(int n)
{
    return new double[n * n]();   // zero-initialised
}

void freeMatrix(Matrix m)
{
    delete[] m;
}

// Accessor macro  A[i][j]  ?  A[i*n + j]
#define AT(M, i, j, n)  ((M)[(i)*(n) + (j)])

void fillRandom(Matrix m, int n)
{
    for (int i = 0; i < n * n; i++)
        m[i] = (double)(rand() % FILL_RANGE + 1);
}

// --- Sequential multiplication --------------------------------
void multiplySequential(Matrix A, Matrix B, Matrix C, int n)
{
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)        // k-loop hoisted for better locality
        {
            double aik = AT(A, i, k, n);
            for (int j = 0; j < n; j++)
                AT(C, i, j, n) += aik * AT(B, k, j, n);
        }
}

// --- Parallel multiplication (OpenMP) ------------------------
void multiplyParallel(Matrix A, Matrix B, Matrix C, int n, int numThreads)
{
    omp_set_num_threads(numThreads);

    #pragma omp parallel for schedule(static) shared(A, B, C, n)
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
        {
            double aik = AT(A, i, k, n);
            for (int j = 0; j < n; j++)
                AT(C, i, j, n) += aik * AT(B, k, j, n);
        }
}

// --- Verify results are identical ----------------------------
bool verifyResults(Matrix Cs, Matrix Cp, int n, double tol = 1e-6)
{
    for (int i = 0; i < n * n; i++)
        if (fabs(Cs[i] - Cp[i]) > tol)
            return false;
    return true;
}

// --- Pretty print a small matrix -----------------------------
void printMatrix(Matrix m, int n, const char* name)
{
    cout << "\nMatrix " << name << " (" << n << "x" << n << "):\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << setw(8) << fixed << setprecision(1) << AT(m, i, j, n);
        cout << "\n";
    }
}

// --- Separator ------------------------------------------------
void separator()
{
    cout << "\n" << string(60, '=') << "\n";
}

// -------------------------------------------------------------
//  MENU OPTION 1: Single Run with user-chosen N and threads
// -------------------------------------------------------------
void singleRun()
{
    int n, numThreads;

    cout << "\nEnter matrix size N (NxN, max " << MAX_N << "): ";
    cin >> n;
    if (n <= 0 || n > MAX_N)
    {
        cout << "Invalid size. Returning to menu.\n";
        return;
    }

    cout << "Enter number of OpenMP threads: ";
    cin >> numThreads;
    if (numThreads <= 0)
    {
        cout << "Invalid thread count. Returning to menu.\n";
        return;
    }

    int maxAvail = omp_get_max_threads();
    if (numThreads > maxAvail)
    {
        cout << "Warning: system supports " << maxAvail
             << " threads. Capping to " << maxAvail << ".\n";
        numThreads = maxAvail;
    }

    cout << "\nAllocating matrices...\n";
    Matrix A  = allocMatrix(n);
    Matrix B  = allocMatrix(n);
    Matrix Cs = allocMatrix(n);   // sequential result
    Matrix Cp = allocMatrix(n);   // parallel result

    srand((unsigned)time(NULL));
    fillRandom(A, n);
    fillRandom(B, n);

    if (n <= 8)
    {
        printMatrix(A, n, "A");
        printMatrix(B, n, "B");
    }

    // -- Sequential -------------------------------------------
    cout << "\nRunning sequential multiplication...\n";
    double tSeqStart = omp_get_wtime();
    multiplySequential(A, B, Cs, n);
    double tSeqEnd   = omp_get_wtime();
    double seqTime   = tSeqEnd - tSeqStart;

    // -- Parallel ---------------------------------------------
    cout << "Running parallel multiplication (" << numThreads << " threads)...\n";
    double tParStart = omp_get_wtime();
    multiplyParallel(A, B, Cp, n, numThreads);
    double tParEnd   = omp_get_wtime();
    double parTime   = tParEnd - tParStart;

    // -- Results ----------------------------------------------
    if (n <= 8)
        printMatrix(Cs, n, "C (result)");

    bool ok = verifyResults(Cs, Cp, n);

    separator();
    cout << fixed << setprecision(6);
    cout << "  Matrix Size      : " << n << " x " << n << "\n";
    cout << "  Threads Used     : " << numThreads << "\n";
    cout << "  Sequential Time  : " << seqTime << " seconds\n";
    cout << "  Parallel Time    : " << parTime   << " seconds\n";

    if (parTime > 0.0)
    {
        double speedup    = seqTime / parTime;
        double efficiency = (speedup / numThreads) * 100.0;
        cout << "  Speedup          : " << fixed << setprecision(3) << speedup << "x\n";
        cout << "  Efficiency       : " << fixed << setprecision(2) << efficiency << "%\n";
    }
    else
    {
        cout << "  Speedup          : N/A (parallel too fast to measure)\n";
    }

    cout << "  Result Verified  : " << (ok ? "PASS" : "FAIL") << "\n";
    separator();

    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(Cs);
    freeMatrix(Cp);
}

// -------------------------------------------------------------
//  MENU OPTION 2: Benchmark across multiple sizes & threads
// -------------------------------------------------------------
void benchmark()
{
    // Sizes and thread counts to test
    int sizes[]   = {100, 250, 500, 750, 1000};
    int nSizes    = 5;
    int threads[] = {1, 2, 4, 8};
    int nThreads  = 4;

    int maxAvail = omp_get_max_threads();
    cout << "\nSystem has " << maxAvail << " logical cores available.\n";
    cout << "Running benchmark (this may take a few minutes for large sizes)...\n";

    separator();
    cout << left
         << setw(8)  << "N"
         << setw(10) << "Threads"
         << setw(16) << "Seq Time(s)"
         << setw(16) << "Par Time(s)"
         << setw(12) << "Speedup"
         << setw(12) << "Efficiency"
         << "\n";
    cout << string(74, '-') << "\n";

    for (int si = 0; si < nSizes; si++)
    {
        int n = sizes[si];

        Matrix A  = allocMatrix(n);
        Matrix B  = allocMatrix(n);
        Matrix Cs = allocMatrix(n);

        srand(42);
        fillRandom(A, n);
        fillRandom(B, n);

        // Sequential time (computed once per size)
        double tS = omp_get_wtime();
        multiplySequential(A, B, Cs, n);
        double seqTime = omp_get_wtime() - tS;

        // Parallel tests
        for (int ti = 0; ti < nThreads; ti++)
        {
            int t = threads[ti];
            if (t > maxAvail) { t = maxAvail; }

            Matrix Cp = allocMatrix(n);

            double tP = omp_get_wtime();
            multiplyParallel(A, B, Cp, n, t);
            double parTime = omp_get_wtime() - tP;

            double speedup    = (parTime > 0) ? seqTime / parTime : 0;
            double efficiency = (t > 0)       ? (speedup / t) * 100.0 : 0;

            cout << fixed
                 << setw(8)  << n
                 << setw(10) << t
                 << setw(16) << setprecision(6) << seqTime
                 << setw(16) << setprecision(6) << parTime
                 << setw(12) << setprecision(3)  << speedup
                 << setw(11) << setprecision(2)  << efficiency << "%"
                 << "\n";

            freeMatrix(Cp);
        }

        freeMatrix(A);
        freeMatrix(B);
        freeMatrix(Cs);
        cout << string(74, '-') << "\n";
    }
}

// -------------------------------------------------------------
//  MENU OPTION 3: Custom thread sweep for one matrix size
// -------------------------------------------------------------
void threadSweep()
{
    int n;
    cout << "\nEnter matrix size N for thread sweep (max " << MAX_N << "): ";
    cin >> n;
    if (n <= 0 || n > MAX_N)
    {
        cout << "Invalid size.\n";
        return;
    }

    int maxT = omp_get_max_threads();
    cout << "Will sweep threads from 1 to " << maxT << " for " << n << "x" << n << " matrix.\n\n";

    Matrix A  = allocMatrix(n);
    Matrix B  = allocMatrix(n);
    Matrix Cs = allocMatrix(n);

    srand((unsigned)time(NULL));
    fillRandom(A, n);
    fillRandom(B, n);

    // Baseline sequential
    double tS = omp_get_wtime();
    multiplySequential(A, B, Cs, n);
    double seqTime = omp_get_wtime() - tS;

    cout << "Sequential time: " << fixed << setprecision(6) << seqTime << " s\n\n";
    cout << left
         << setw(10) << "Threads"
         << setw(16) << "Par Time(s)"
         << setw(12) << "Speedup"
         << setw(14) << "Efficiency"
         << "\n";
    cout << string(52, '-') << "\n";

    for (int t = 1; t <= maxT; t++)
    {
        Matrix Cp = allocMatrix(n);

        double tP = omp_get_wtime();
        multiplyParallel(A, B, Cp, n, t);
        double parTime = omp_get_wtime() - tP;

        double speedup    = (parTime > 0) ? seqTime / parTime : 0;
        double efficiency = (speedup / t) * 100.0;

        cout << setw(10) << t
             << setw(16) << fixed << setprecision(6) << parTime
             << setw(12) << setprecision(3)  << speedup
             << setw(12) << setprecision(2)  << efficiency << "%"
             << "\n";

        freeMatrix(Cp);
    }

    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(Cs);
}

// --- Main -----------------------------------------------------
int main()
{
    cout << "============================================================\n";
    cout << "    Parallel Matrix Multiplication using OpenMP\n";
    cout << "    Parallel and Distributed Computing - Project\n";
    cout << "============================================================\n";
    cout << "  Group Members:\n";
    cout << "    Muhammad Ali Haris       (22k-4239)\n";
    cout << "    Muhammad Abdullah Shariq (22k-4497)\n";
    cout << "    Hamza Haroon             (22k-4200)\n";
    cout << "============================================================\n";
    cout << "  OpenMP Version : " << _OPENMP << "\n";
    cout << "  Max Threads    : " << omp_get_max_threads() << "\n";
    cout << "============================================================\n";

    int choice;
    do
    {
        cout << "\n--- MAIN MENU ---\n";
        cout << "  1. Single Run   (custom N and thread count)\n";
        cout << "  2. Benchmark    (multiple sizes and thread counts)\n";
        cout << "  3. Thread Sweep (one size, sweep all thread counts)\n";
        cout << "  4. Exit\n";
        cout << "Choose option: ";
        cin >> choice;

        switch (choice)
        {
            case 1: singleRun();   break;
            case 2: benchmark();   break;
            case 3: threadSweep(); break;
            case 4: cout << "\nExiting. Goodbye!\n"; break;
            default: cout << "Invalid choice. Please try again.\n";
        }
    }
    while (choice != 4);

    return 0;
}