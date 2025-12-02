#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>
#include <omp.h> 

#define M_PI (4.0 * std::atan(1.0))

// Ogranicavamo dubinu paralelizacije
// Posto se svaka grana dijeli na 2, dubina 4 znaci do 16 niti (2^4)
// Ako ima 8 jezgara, dubina 3 je optimalna
#define MAX_PARALLEL_DEPTH 3

using namespace std;

struct Point {
    double x, y;
};

double cross(const Point& A, const Point& B, const Point& C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

double distance(const Point& A, const Point& B, const Point& C) {
    return fabs(cross(A, B, C));
}

// Ovu funkciju cemo pozivati serijski unutar rekurzije
int farthestPoint(const vector<Point>& pts, const Point& A, const Point& B) {
    double maxDist = -1.0;
    int idx = -1;
    for (int i = 0; i < pts.size(); i++) {
        double d = distance(A, B, pts[i]);
        if (d > maxDist) {
            maxDist = d;
            idx = i;
        }
    }
    return idx;
}

// Rekurzivna funkcija sa 'depth' parametrom za kontrolu OpenMP sekcija
vector<Point> quickHullRecursive(const vector<Point>& pts, const Point& A, const Point& B, int depth) {
    vector<Point> resultHull;

    int idx = farthestPoint(pts, A, B);

    if (idx == -1) {
        resultHull.push_back(B);
        return resultHull;
    }

    Point P = pts[idx];

    vector<Point> leftAP, leftPB;
    leftAP.reserve(pts.size() / 2);
    leftPB.reserve(pts.size() / 2);

    for (const Point& X : pts) {
        if (cross(A, P, X) > 0)
            leftAP.push_back(X);
        else if (cross(P, B, X) > 0)
            leftPB.push_back(X);
    }

    vector<Point> hullLeft, hullRight;

    // Ako nismo premasili dozvoljenu dubinu i imamo dovoljno tacaka, dijelimo posao na niti
    if (depth < MAX_PARALLEL_DEPTH && pts.size() > 20000) {
        // 'parallel sections' omogucava da razlicite niti rade razlicite blokove
#pragma omp parallel sections
        {
#pragma omp section
            {
                hullLeft = quickHullRecursive(leftAP, A, P, depth + 1);
            }

#pragma omp section
            {
                hullRight = quickHullRecursive(leftPB, P, B, depth + 1);
            }
        }
    }
    else {
        // Ako smo duboko u rekurziji, nastavljamo serijski (brze je nego praviti nove niti)
        hullLeft = quickHullRecursive(leftAP, A, P, depth + 1);
        hullRight = quickHullRecursive(leftPB, P, B, depth + 1);
    }

    // Spajanje rezultata (ovo mora serijski)
    resultHull.insert(resultHull.end(), hullLeft.begin(), hullLeft.end());
    resultHull.insert(resultHull.end(), hullRight.begin(), hullRight.end());

    if (resultHull.size() > 1 && resultHull.back().x == resultHull.front().x && resultHull.back().y == resultHull.front().y) resultHull.pop_back();   //////


    return resultHull;
}

vector<Point> quickHull2D(const vector<Point>& pts) {
    if (pts.size() < 3) return pts;

    int minX = 0, maxX = 0;
    double minVal = pts[0].x;
    double maxVal = pts[0].x;

    for (int i = 1; i < pts.size(); i++) {
        if (pts[i].x < pts[minX].x) minX = i;
        if (pts[i].x > pts[maxX].x) maxX = i;
    }

    Point A = pts[minX];
    Point B = pts[maxX];

    vector<Point> leftSet, rightSet;
    leftSet.reserve(pts.size());
    rightSet.reserve(pts.size());

    // Podjela na gornji i donji skup
    for (int i = 0; i < pts.size(); i++) {
        if (i == minX || i == maxX) continue;
        if (cross(A, B, pts[i]) > 0) leftSet.push_back(pts[i]);
        else rightSet.push_back(pts[i]);
    }

    vector<Point> finalHull;
    finalHull.push_back(A);

    vector<Point> h1, h2;

    // Prvi nivo paralelizacije: Jedna nit radi gornji dio, druga donji dio

#pragma omp parallel sections
    {
#pragma omp section
        {
            h1 = quickHullRecursive(leftSet, A, B, 1);
        }
#pragma omp section
        {
            h2 = quickHullRecursive(rightSet, B, A, 1);
        }
    }

    finalHull.insert(finalHull.end(), h1.begin(), h1.end());
    finalHull.insert(finalHull.end(), h2.begin(), h2.end());

    return finalHull;
}

int main() {
    // Moramo eksplicitno dozvoliti da nit napravi nove niti 
    omp_set_nested(1);

    // najgori slucaj
    const int N = 10000000;
    const double MIN_R = 200.0; // Minimalna udaljenost od centra
    const double MAX_R = 500.0; // Maksimalna udaljenost od centra

    double minVrijeme = 1e9;
    double maxVrijeme = 0.0;
    double ukupnoVrijeme = 0.0;
    int brojIteracija = 1;

    for (int i = 0; i < brojIteracija; i++) {
        std::vector<Point> tacke;
        tacke.reserve(N);
        // Priprema generatora nasumicnih brojeva
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        // Definisemo opseg za R (od MIN_R do MAX_R)
        std::uniform_real_distribution<double> dist_r(MIN_R, MAX_R);
        // Za svaku tacku generisemo novi, nasumicni radijus
        double current_R = dist_r(rng);
        //std::cout << "Izabran radijus za ovu iteraciju: " << current_R << "\n";
        for (int i = 0; i < N; i++) {
            // Ugao ostaje pravilan (kako bi tacke isle u krug)
            double angle = (2 * M_PI * i) / N;
            double x = current_R * cos(angle);
            double y = current_R * sin(angle);
            tacke.push_back({ x, y });
        }
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Point> hull = quickHull2D(tacke);
        auto end = std::chrono::high_resolution_clock::now();
        double trajanje = std::chrono::duration<double, std::milli>(end - start).count();
        //std::cout << "Generisano: " << N << " random tacaka\n";
        //std::cout << "Hull ima: " << hull.size() << " tacaka\n";
        //std::cout << "Vrijeme: " << trajanje << " ms\n";

        if (trajanje < minVrijeme) minVrijeme = trajanje;
        if (trajanje > maxVrijeme) maxVrijeme = trajanje;
        ukupnoVrijeme += trajanje;
    }

    // Ispis rezultata
    std::cout << "\n================ REZULTATI ================\n";
    std::cout << "Broj iteracija: " << brojIteracija << endl;
    std::cout << "Broj tacaka:    " << N << endl;
    std::cout << "-------------------------------------------\n";
    std::cout << "MIN vrijeme:      " << minVrijeme << " ms\n";
    std::cout << "MAX vrijeme:      " << maxVrijeme << " ms\n";
    std::cout << "PROSJECNO vrijeme: " << ukupnoVrijeme / brojIteracija << " ms\n";
    std::cout << "===========================================\n";


    return 0;
}
