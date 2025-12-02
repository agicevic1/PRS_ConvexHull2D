#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>
#include <immintrin.h> 

#define M_PI (4.0 * std::atan(1.0))

using namespace std;

struct Point {
    double x, y;
};

// skalarni cross proizvod 
inline double cross(const Point& A, const Point& B, const Point& C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

// SIMD optimizovana funkcija za pronalazak najudaljenije tacke
int farthestPointAVX(const vector<Point>& pts, const Point& A, const Point& B) {
    int maxIdx = -1;
    double maxDist = -1.0;

    size_t n = pts.size();
    size_t i = 0;

    // Priprema konstanti za vektorizaciju
    // Formula cross produkta: (Bx-Ax)*(Py-Ay) - (By-Ay)*(Px-Ax)
    double d_BA_x = B.x - A.x;
    double d_BA_y = B.y - A.y;

    // Popunjavamo registre istim vrijednostima
    __m256d v_BA_x = _mm256_set1_pd(d_BA_x);
    __m256d v_BA_y = _mm256_set1_pd(d_BA_y);
    __m256d v_A_x = _mm256_set1_pd(A.x);
    __m256d v_A_y = _mm256_set1_pd(A.y);

    // Maska za apsolutnu vrijednost (brisanje znaka)
    __m256d v_sign_mask = _mm256_set1_pd(-0.0);

    // Trenutni maksimum u vektorskom obliku (sve nule na pocetku)
    __m256d v_maxDist = _mm256_set1_pd(-1.0);

    // Procesiramo 4 tacke odjednom (4 * 2 double-a = 256 bita po X i Y)
    // Struktura u memoriji je: x0 y0 x1 y1 x2 y2 x3 y3 ...
    for (; i + 3 < n; i += 4) {
        // Ucita 4 tacke (8 double-ova) u dva registra
        __m256d r1 = _mm256_loadu_pd(&pts[i].x);     // x0 y0 x1 y1
        __m256d r2 = _mm256_loadu_pd(&pts[i + 2].x); // x2 y2 x3 y3

        // Raspakivanje: Trebaju nam svi X-ovi u jednom, svi Y-ovi u drugom registru
        // 1. Permutacija 128-bitnih blokova: r1_low, r2_low -> x0 y0 x2 y2
        __m256d t1 = _mm256_permute2f128_pd(r1, r2, 0x20);
        // 2. Permutacija: r1_high, r2_high -> x1 y1 x3 y3
        __m256d t2 = _mm256_permute2f128_pd(r1, r2, 0x31);

        // 3. Unpack: dobijamo ciste X i Y vektore
        __m256d v_Px = _mm256_unpacklo_pd(t1, t2); // x0 x1 x2 x3
        __m256d v_Py = _mm256_unpackhi_pd(t1, t2); // y0 y1 y2 y3

        // Racunanje: dy = (Py - Ay)
        __m256d v_diff_y = _mm256_sub_pd(v_Py, v_A_y);
        // Racunanje: dx = (Px - Ax)
        __m256d v_diff_x = _mm256_sub_pd(v_Px, v_A_x);

        // Term 1: (Bx - Ax) * dy
        __m256d term1 = _mm256_mul_pd(v_BA_x, v_diff_y);
        // Term 2: (By - Ay) * dx
        __m256d term2 = _mm256_mul_pd(v_BA_y, v_diff_x);

        // Cross product: term1 - term2
        __m256d v_cross = _mm256_sub_pd(term1, term2);

        // Absolutna vrijednost (Distance): andnot sa maskom znaka
        __m256d v_dist = _mm256_andnot_pd(v_sign_mask, v_cross);

        // Optimizacija: Provjeravamo da li je BILO KOJA od 4 distance veca od trenutnog globalnog max-a
        // _mm256_cmp_pd vraca masku bitova
        __m256d v_cmp = _mm256_cmp_pd(v_dist, v_maxDist, _CMP_GT_OQ);

        // movemask prebacuje sign bitove u int. Ako je != 0, nasli smo novi maksimum.
        if (_mm256_movemask_pd(v_cmp)) {
            // Izvlacenje rezultata u privremeni niz da azuriramo max
            double res[4];
            _mm256_storeu_pd(res, v_dist);

            for (int k = 0; k < 4; ++k) {
                if (res[k] > maxDist) {
                    maxDist = res[k];
                    maxIdx = i + k;
                    // Azuriramo i vektorski max 
                    v_maxDist = _mm256_set1_pd(maxDist);
                }
            }
        }
    }

    // Ostatak (ako broj tacaka nije deljiv sa 4, rjesavamo skalarno)
    for (; i < n; i++) {
        double d = std::abs(cross(A, B, pts[i]));
        if (d > maxDist) {
            maxDist = d;
            maxIdx = i;
        }
    }

    return maxIdx;
}

// Rekurzivni QuickHull
void quickHull(const vector<Point>& pts, const Point& A, const Point& B,
    vector<Point>& hull) {

    // KORISTIMO NOVU SIMD FUNKCIJU
    int idx = farthestPointAVX(pts, A, B);

    if (idx == -1) {
        hull.push_back(B);
        return;
    }

    Point P = pts[idx];

    vector<Point> leftAP, leftPB;
    leftAP.reserve(pts.size() / 2); 
    leftPB.reserve(pts.size() / 2);

    for (const Point& X : pts) {
        double c1 = cross(A, P, X);
        if (c1 > 0) {
            leftAP.push_back(X);
            continue;
        }

        // Ako nije lijevo od AP, tek onda proveravamo PB
        double c2 = cross(P, B, X);
        if (c2 > 0)
            leftPB.push_back(X);
    }

    quickHull(leftAP, A, P, hull);
    quickHull(leftPB, P, B, hull);

}

// Glavna funkcija: vraca konveksni omotac
vector<Point> quickHull2D(const vector<Point>& pts) {
    vector<Point> hull;
    if (pts.size() < 3)
        return pts;

    // Pronadji krajnju lijevu i krajnju desnu tacku
    int minX = 0, maxX = 0;
    for (int i = 1; i < pts.size(); i++) {
        if (pts[i].x < pts[minX].x) minX = i;
        if (pts[i].x > pts[maxX].x) maxX = i;
    }

    Point A = pts[minX];
    Point B = pts[maxX];

    vector<Point> leftSet, rightSet;
    leftSet.reserve(pts.size());
    rightSet.reserve(pts.size());

    for (int i = 0; i < pts.size(); i++) {
        if (i == minX || i == maxX) continue;

        if (cross(A, B, pts[i]) > 0)
            leftSet.push_back(pts[i]);
        else
            rightSet.push_back(pts[i]);
    }

    hull.push_back(A);
    quickHull(leftSet, A, B, hull);
    quickHull(rightSet, B, A, hull);

    return hull;
}

int main() {
    const int N = 10000000;
    const double MIN_R = 200.0;
    const double MAX_R = 500.0;

    double minVrijeme = 1e9;
    double maxVrijeme = 0.0;
    double ukupnoVrijeme = 0.0;
    int brojIteracija = 1; 

    for (int iter = 0; iter < brojIteracija; iter++) {
        std::vector<Point> tacke;
        tacke.reserve(N);
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> dist_r(MIN_R, MAX_R);
        double current_R = dist_r(rng);

        for (int i = 0; i < N; i++) {
            double angle = (2 * M_PI * i) / N;
            double x = current_R * cos(angle);
            double y = current_R * sin(angle);
            tacke.push_back({ x, y });
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Point> hull = quickHull2D(tacke);
        auto end = std::chrono::high_resolution_clock::now();

        double trajanje = std::chrono::duration<double, std::milli>(end - start).count();

        if (trajanje < minVrijeme) minVrijeme = trajanje;
        if (trajanje > maxVrijeme) maxVrijeme = trajanje;
        ukupnoVrijeme += trajanje;

        //std::cout << "Iteracija " << iter + 1 << ": " << trajanje << " ms" << std::endl;
    }

    std::cout << "\n================ REZULTATI (SIMD AVX2) ================\n";
    std::cout << "Broj iteracija: " << brojIteracija << endl;
    std::cout << "Broj tacaka:    " << N << endl;
    std::cout << "-------------------------------------------\n";
    std::cout << "MIN vrijeme:      " << minVrijeme << " ms\n";
    std::cout << "MAX vrijeme:      " << maxVrijeme << " ms\n";
    std::cout << "PROSJECNO vrijeme: " << ukupnoVrijeme / brojIteracija << " ms\n";
    std::cout << "===========================================\n";

    return 0;
}