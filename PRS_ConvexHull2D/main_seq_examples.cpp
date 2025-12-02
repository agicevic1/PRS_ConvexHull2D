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

// Primjeri koristenja
int main() {

    //Input: points[] = { {0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3} };
    //Output:  The points in convex hull are : (0, 0) (0, 3) (3, 1) (4, 4)
    vector<Point> t1 = { {0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3} };
    vector<Point> h1 = quickHull2D(t1);
    cout << "Konveksni omotac1: " << endl;
    for (auto& p : h1)
        cout << "(" << p.x << ", " << p.y << ")\n";

    // Input primjer 2
    // Input: points[] = { (16, 3), (12, 17), (0, 6), (-4, -6), (16, 6), (16, -7), (16, -3),
    //              (17, -4), (5, 19), (19, -8), (3, 16), (12, 13), (3, -4), (17, 5),
    //              (-3, 15), (-3, -9), (0, 11), (-9, -3), (-4, -2), (12, 10) }
    // Output convex hull : (-9, -3), (-3, -9), (19, -8), (17, 5), (12, 17), (5, 19), (-3, 15)
    vector<Point> t2 = { {16, 3}, {12, 17}, {0, 6}, {-4, -6}, {16, 6}, {16, -7},
                         {16, -3}, {17, -4}, {5, 19}, {19, -8}, {3, 16}, {12, 13},
                         {3, -4}, {17, 5}, {-3, 15}, {-3, -9}, {0, 11}, {-9, -3},
                         {-4, -2}, {12, 10} };
    vector<Point> h2 = quickHull2D(t2);
    cout << "Konveksni omotac2: " << endl;
    for (auto& p : h2)
        cout << "(" << p.x << ", " << p.y << ")\n";

    return 0;
}