#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>    
#include <ctime>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

using namespace std;

const double PI = 4.0 * atan(1.0);
const double EPSILON = 1e-13;

// Dubina 4 ce generisati do 16 (2^4) nezavisnih zadataka
const int DECOMPOSITION_DEPTH = 4;

struct Point {
    double x, y;
};

// Struktura koja opisuje jedan posao za thread
struct HullTask {
    Point* startPtr;
    Point* endPtr;
    Point p1;
    Point p2;
};

inline double crossProduct(const Point& a, const Point& b, const Point& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

#include <immintrin.h>

inline Point* findMaxSimd(Point* startPtr, Point* endPtr,
    const Point& p1, const Point& p2, double& out_max)
{
    __m256d ax = _mm256_set1_pd(p1.x);
    __m256d ay = _mm256_set1_pd(p1.y);
    __m256d bx = _mm256_set1_pd(p2.x);
    __m256d by = _mm256_set1_pd(p2.y);

    __m256d dx = _mm256_sub_pd(bx, ax);
    __m256d dy = _mm256_sub_pd(by, ay);

    __m256d maxVal = _mm256_set1_pd(-1e300);
    __m256i maxIndex = _mm256_setzero_si256();

    Point* base = startPtr;
    int index = 0;

    for (Point* curr = startPtr; curr + 4 <= endPtr; curr += 4, index += 4)
    {
        __m256d px = _mm256_set_pd(curr[3].x, curr[2].x, curr[1].x, curr[0].x);
        __m256d py = _mm256_set_pd(curr[3].y, curr[2].y, curr[1].y, curr[0].y);

        __m256d px2 = _mm256_sub_pd(px, ax);
        __m256d py2 = _mm256_sub_pd(py, ay);

        __m256d cross1 = _mm256_mul_pd(dx, py2);
        __m256d cross2 = _mm256_mul_pd(dy, px2);
        __m256d cross = _mm256_sub_pd(cross1, cross2);

        __m256d cmp = _mm256_cmp_pd(cross, maxVal, _CMP_GT_OQ);

        maxVal = _mm256_blendv_pd(maxVal, cross, cmp);

        __m256i idx = _mm256_set_epi64x(index + 3, index + 2, index + 1, index);
        maxIndex = _mm256_blendv_epi8(maxIndex, idx, _mm256_castpd_si256(cmp));
    }

    double vals[4];
    long long idxs[4];

    _mm256_storeu_pd(vals, maxVal);
    _mm256_storeu_si256((__m256i*)idxs, maxIndex);

    double finalMax = -1e300;
    long long best = -1;

    for (int i = 0; i < 4; i++)
        if (vals[i] > finalMax) {
            finalMax = vals[i];
            best = idxs[i];
        }

    Point* bestPtr = nullptr;
    if (best >= 0) {
        bestPtr = base + best;
        out_max = finalMax;
    }

    for (Point* curr = startPtr + ((endPtr - startPtr) & ~3ULL);
        curr < endPtr; curr++)
    {
        double d = crossProduct(p1, p2, *curr);
        if (d > out_max) {
            out_max = d;
            bestPtr = curr;
        }
    }

    return bestPtr;
}


// --- SEKVENCIJALNA VERZIJA (worker) ---
void findHullSeq(Point* startPtr, Point* endPtr, const Point p1, const Point p2, vector<Point>& output) {
    if (startPtr >= endPtr) return;

    double max_dist = -1.0;
    Point* max_ptr = findMaxSimd(startPtr, endPtr, p1, p2, max_dist);


    if (max_ptr == nullptr || max_dist < EPSILON) return;

    Point c = *max_ptr;
    output.push_back(c);

    std::iter_swap(startPtr, max_ptr);
    Point* workerStart = startPtr + 1;

    Point* left_pivot = workerStart;
    for (Point* curr = workerStart; curr < endPtr; ++curr) {
        if (crossProduct(p1, c, *curr) > EPSILON) {
            std::iter_swap(curr, left_pivot);
            left_pivot++;
        }
    }

    findHullSeq(workerStart, left_pivot, p1, c, output);

    Point* right_pivot = left_pivot;
    for (Point* curr = left_pivot; curr < endPtr; ++curr) {
        if (crossProduct(c, p2, *curr) > EPSILON) {
            std::iter_swap(curr, right_pivot);
            right_pivot++;
        }
    }

    findHullSeq(left_pivot, right_pivot, c, p2, output);
}

// --- DEKOMPOZICIJA ---
// Ova funkcija ne rjesava problem do kraja, vec ga dijeli na manje dijelove i puni vektor 'tasks'
void decomposeRecursive(Point* startPtr, Point* endPtr, const Point p1, const Point p2, int depth, vector<HullTask>& tasks) {
    if (startPtr >= endPtr) return;

    // Ako smo dosegli zeljenu dubinu, prekidamo
    if (depth >= DECOMPOSITION_DEPTH) {
        tasks.push_back({ startPtr, endPtr, p1, p2 });
        return;
    }

    double max_dist = -1.0;
    Point* max_ptr = findMaxSimd(startPtr, endPtr, p1, p2, max_dist);


    if (max_ptr == nullptr || max_dist < EPSILON) return;

    Point c = *max_ptr;
    // Trazimo C tacku, ali je NE ubacujemo u rezultat jos (ubacice je worker thread)

    std::iter_swap(startPtr, max_ptr);
    Point* workerStart = startPtr + 1;

    Point* left_pivot = workerStart;
    for (Point* curr = workerStart; curr < endPtr; ++curr) {
        if (crossProduct(p1, c, *curr) > EPSILON) {
            std::iter_swap(curr, left_pivot);
            left_pivot++;
        }
    }

    Point* right_pivot = left_pivot;
    for (Point* curr = left_pivot; curr < endPtr; ++curr) {
        if (crossProduct(c, p2, *curr) > EPSILON) {
            std::iter_swap(curr, right_pivot);
            right_pivot++;
        }
    }

    decomposeRecursive(workerStart, left_pivot, p1, c, depth + 1, tasks);
    decomposeRecursive(left_pivot, right_pivot, c, p2, depth + 1, tasks);
}

// Modifikovani decompose koji vraca i tacke nadjene usput
void decomposeAndCollect(Point* startPtr, Point* endPtr, const Point p1, const Point p2, int depth, vector<HullTask>& tasks, vector<Point>& pre_hull) {
    if (startPtr >= endPtr) return;

    if (depth >= DECOMPOSITION_DEPTH) {
        tasks.push_back({ startPtr, endPtr, p1, p2 });
        return;
    }

    double max_dist = -1.0;
    Point* max_ptr = findMaxSimd(startPtr, endPtr, p1, p2, max_dist);


    if (max_ptr == nullptr || max_dist < EPSILON) return;

    Point c = *max_ptr;
    pre_hull.push_back(c); // Cuvamo tacku odmah

    std::iter_swap(startPtr, max_ptr);
    Point* workerStart = startPtr + 1;

    Point* left_pivot = workerStart;
    for (Point* curr = workerStart; curr < endPtr; ++curr) {
        if (crossProduct(p1, c, *curr) > EPSILON) {
            std::iter_swap(curr, left_pivot);
            left_pivot++;
        }
    }

    Point* right_pivot = left_pivot;
    for (Point* curr = left_pivot; curr < endPtr; ++curr) {
        if (crossProduct(c, p2, *curr) > EPSILON) {
            std::iter_swap(curr, right_pivot);
            right_pivot++;
        }
    }

    decomposeAndCollect(workerStart, left_pivot, p1, c, depth + 1, tasks, pre_hull);
    decomposeAndCollect(left_pivot, right_pivot, c, p2, depth + 1, tasks, pre_hull);
}


void quickHull(vector<Point>& points, vector<Point>& final_hull) {
    final_hull.clear();
    size_t n = points.size();
    if (n < 3) return;

    final_hull.reserve(n);

    // 1. Min/Max X
    size_t min_idx = 0;
    size_t max_idx = 0;
    for (size_t i = 1; i < n; i++) {
        if (points[i].x < points[min_idx].x) min_idx = i;
        if (points[i].x > points[max_idx].x) max_idx = i;
    }

    Point pA = points[min_idx];
    Point pB = points[max_idx];
    final_hull.push_back(pA);
    final_hull.push_back(pB);

    std::swap(points[0], points[min_idx]);
    if (max_idx == 0) max_idx = min_idx;
    std::swap(points[1], points[max_idx]);

    Point* start = &points[0];
    Point* workerStart = start + 2;
    Point* end = start + n;

    // Inicijalna podjela
    Point* split_point = workerStart;
    for (Point* curr = workerStart; curr < end; ++curr) {
        if (crossProduct(pA, pB, *curr) > EPSILON) {
            std::iter_swap(curr, split_point);
            split_point++;
        }
    }

    Point* split_point_2 = split_point;
    for (Point* curr = split_point; curr < end; ++curr) {
        if (crossProduct(pA, pB, *curr) < -EPSILON) {
            std::iter_swap(curr, split_point_2);
            split_point_2++;
        }
    }

    // --- PRIPREMA ZADATAKA, pravi raspored za threadove
    vector<HullTask> tasks;
    tasks.reserve(64);

    // Gornji dio
    decomposeAndCollect(workerStart, split_point, pA, pB, 1, tasks, final_hull);
    // Donji dio
    decomposeAndCollect(split_point, split_point_2, pB, pA, 1, tasks, final_hull);

    // --- PARALELNO IZVRSAVANJE (DYNAMIC SCHEDULE) ---
    // Cim thread zavrsi posao, uzima sljedeci, nema cekanja

    int num_tasks = (int)tasks.size();

    // Vektor vektora da svaki thread pise u svoje
    // (OpenMP nema direktan thread-local vector, pa improvizujemo nizom)
    int max_threads = omp_get_max_threads();
    vector<vector<Point>> thread_results(max_threads);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_tasks; ++i) {
        int tid = omp_get_thread_num();
        HullTask& t = tasks[i];

        // Svaki thread radi svoj zadatak sekvencijalno
        findHullSeq(t.startPtr, t.endPtr, t.p1, t.p2, thread_results[tid]);
    }

    // --- SPAJANJE REZULTATA ---
    for (int i = 0; i < max_threads; ++i) {
        final_hull.insert(final_hull.end(), thread_results[i].begin(), thread_results[i].end());
    }
}

int main() {
    //omp_set_nested(0);

    omp_set_num_threads(8);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 2.0 * PI);

    int N = 10000000;
    double radius = 10000000.0;

    cout << "Generisanje " << N << " tacaka..." << endl;
    vector<Point> points;
    points.reserve(N);

    for (int i = 0; i < N; i++) {
        double angle = dis(gen);
        points.push_back({ radius * cos(angle), radius * sin(angle) });
    }

    vector<Point> result_hull;

    auto start = std::chrono::high_resolution_clock::now();

    quickHull(points, result_hull);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double, std::milli>(end - start).count();

    cout << "Vrijeme: " << elapsed_time << " ms" << endl;
    cout << "Tacaka u omotacu: " << result_hull.size() << endl;

    return 0;
}