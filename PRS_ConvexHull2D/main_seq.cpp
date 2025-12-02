#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>

#define M_PI (4.0 * std::atan(1.0))

using namespace std;

struct Point {
    double x, y;
};

// Vektorski proizvod (determinanta)
// det > 0 C je lijevo od AB, det < 0 C je desno, det = 0 C je na liniji
double cross(const Point& A, const Point& B, const Point& C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

// Udaljenost tacke od pravca AB (apsolutna vrijednost determinante)
double distance(const Point& A, const Point& B, const Point& C) {
    return fabs(cross(A, B, C));
}

// Pronalazi tacku najudaljeniju od pravca AB
int farthestPoint(const vector<Point>& pts, const Point& A, const Point& B) {
    double maxDist = 0;
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

// Rekurzivni QuickHull
void quickHull(const vector<Point>& pts, const Point& A, const Point& B,
    vector<Point>& hull) {

    int idx = farthestPoint(pts, A, B);
    if (idx == -1) {
        hull.push_back(B);
        return;
    }

    Point P = pts[idx];

    vector<Point> leftAP, leftPB;
    leftAP.reserve(pts.size());
    leftPB.reserve(pts.size());

    for (const Point& X : pts) {
        // odredjujemo je li tacka lijevo od segmenta
        if (cross(A, P, X) > 0)
            leftAP.push_back(X);
        else if (cross(P, B, X) > 0)
            leftPB.push_back(X);
    }

    quickHull(leftAP, A, P, hull);
    quickHull(leftPB, P, B, hull);

    if (hull.size() > 1 && hull.back().x == hull.front().x && hull.back().y == hull.front().y) hull.pop_back();

}

// Glavna funkcija: vraca konveksni omotac
vector<Point> quickHull2D(const vector<Point> &pts) {
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

// Primjer koristenja
int main() {
    
    const int N = 1000000;
    std::vector<Point> tacke;
    tacke.reserve(N);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (int i = 0; i < N; i++) {
        double x = dist(rng);
        double y = dist(rng);
        tacke.push_back({ x, y });
    }
    auto start1 = std::chrono::high_resolution_clock::now();
    std::vector<Point> hull = quickHull2D(tacke);
    auto end1 = std::chrono::high_resolution_clock::now();
    double ms1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
    cout << "Generisano: " << N << " random tacaka\n";
    cout << "Hull ima: " << hull.size() << " tacaka\n";
    cout << "Vrijeme: " << ms1 << " ms\n";
    cout << "Konveksni omotac:" << endl;
    for (auto& p : hull)
        cout << "(" << p.x << ", " << p.y << ")\n";


    return 0;
}