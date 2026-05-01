# 🚀 Parallel Matrix Multiplication using OpenMP

## 📌 Overview

This project implements **matrix multiplication** using both:

* ✅ Sequential approach
* ⚡ Parallel approach using OpenMP

The goal is to **analyze performance improvements** achieved through parallel computing by comparing:

* Execution time
* Speedup
* Efficiency

---

## 🧠 Key Concepts

* Parallel Computing
* OpenMP (Multithreading)
* Matrix Multiplication
* Speedup & Efficiency Analysis
* Cache Optimization (1D Array Representation)

---

## 🏗️ Implementation Details

### 🔹 Matrix Representation

* Matrices are stored as **1D arrays** instead of 2D arrays
* Improves **cache locality and performance**

```cpp
#define AT(M, i, j, n) ((M)[(i)*(n) + (j)])
```

---

### 🔹 Sequential Multiplication

* Standard triple nested loop
* Computes:

[
C[i][j] = \sum_{k=0}^{n-1} A[i][k] \times B[k][j]
]

---

### 🔹 Parallel Multiplication (OpenMP)

* Uses:

```cpp
#pragma omp parallel for
```

* Distributes work across multiple threads
* Improves performance for large matrices

---

### 🔹 Performance Metrics

* **Execution Time** → Measured using `omp_get_wtime()`
* **Speedup**
  [
  Speedup = \frac{T_{sequential}}{T_{parallel}}
  ]
* **Efficiency**
  [
  Efficiency = \frac{Speedup}{Number\ of\ Threads} \times 100
  ]

---

## ⚙️ Features

* Interactive menu-based program:

  * 🔸 Single Run (custom matrix size & threads)
  * 🔸 Benchmark Mode (multiple sizes & threads)
  * 🔸 Thread Sweep (performance vs threads)
* Result verification using tolerance check
* Pretty printing for small matrices

---

## 🖥️ Requirements

* C++ Compiler (GCC recommended)
* OpenMP support enabled

---

## ⚡ How to Run

### 🔧 Using GCC (Recommended)

```bash
g++ main.cpp -o matrix -fopenmp
./matrix
```

---

### 🛠️ Using Dev-C++ (Embarcadero)

1. Go to **Tools → Compiler Options**
2. Add the following flags:

**Compiler:**

```
-fopenmp
```

**Linker:**

```
-fopenmp
```

3. Rebuild and run the project

---

## 📊 Sample Output

```
Matrix Size      : 500 x 500
Threads Used     : 4
Sequential Time  : 2.345 seconds
Parallel Time    : 0.982 seconds
Speedup          : 2.38x
Efficiency       : 59.5%
Result Verified  : PASS
```

---

## ⚠️ Notes

* Performance gain depends on:

  * Number of CPU cores
  * Matrix size
  * Memory bandwidth
* Small matrices may not show significant speedup due to parallel overhead

---

## 🚧 Future Improvements

* Fix race condition in parallel implementation
* Implement cache blocking (tiling)
* Use advanced OpenMP scheduling strategies
* GPU acceleration (CUDA / OpenCL)

---

## 👨‍💻 Authors

* Muhammad Ali Haris (22k-4239)
* Muhammad Abdullah Shariq (22k-4497)
* Hamza Haroon (22k-4200)

---

## 📚 License

This project is for academic purposes.
