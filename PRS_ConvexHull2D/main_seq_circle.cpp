#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <random>    
#include <ctime>     

using namespace std;

const double PI = 4.0 * atan(1.0);
const double EPSILON = 1e-13;

struct Point {
    double x, y;
};

// Globalni kontejner za rezultat (rezervisemo ga unaprijed)
vector<Point> hull;

// inline: Eliminise trosak funkcijskog poziva (function call overhead) za kriticne male operacije koje se izvrsavaju unutar petlji
inline double crossProduct(const Point& a, const Point& b, const Point& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// startPtr: Pokazivac na prvi element niza
// endPtr:   Pokazivac IZA posljednjeg elementa
void findHullOptimized(Point* startPtr, Point* endPtr, const Point p1, const Point p2) {

    // Ako nema tacaka u opsegu, bezi napolje
    if (startPtr >= endPtr) return;

    Point* max_ptr = nullptr;
    double max_dist = -1.0;

    // 1. Trazimo najudaljeniju tacku (Linearni prolaz)
    // Koristimo ++curr (pointer aritmetiku) jer je brza od points[i]
    for (Point* curr = startPtr; curr < endPtr; ++curr) {
        double d = crossProduct(p1, p2, *curr);
        if (d > max_dist) {
            max_dist = d;
            max_ptr = curr;
        }
    }

    // Ako nismo nasli tacku (sve su unutar trougla ili blizu linije)
    if (max_ptr == nullptr || max_dist < EPSILON) return;

    Point c = *max_ptr;
    hull.push_back(c);

    // Najveci problem brzine je provjera "da li je trenutna tacka jednaka C"
    // Rjesenje: Zamijenimo tacku C sa prvom tackom u nizu
    // I onda petlju krecemo od druge tacke. Tako garantovano ne nailazimo na C
    std::iter_swap(startPtr, max_ptr);

    // Nasa radna memorija sada krece od startPtr + 1
    Point* workerStart = startPtr + 1;

    // --- BRZO PARTICIONISANJE (DVA PROLAZA) ---

    // Faza 1: Skupi tacke koje su lijevo od P1 -> C
    // 'left_pivot' je granica dokle smo naslagali dobre tacke
    Point* left_pivot = workerStart;

    for (Point* curr = workerStart; curr < endPtr; ++curr) {

        if (crossProduct(p1, c, *curr) > EPSILON) {
            std::iter_swap(curr, left_pivot);
            left_pivot++;
        }
    }

    // Rekurzija za lijevi trougao
    // Opseg: [workerStart, left_pivot)
    findHullOptimized(workerStart, left_pivot, p1, c);

    // Faza 2: Skupi tacke koje su lijevo od C -> P2
    // Trazimo ih u ostatku niza: od left_pivot do endPtr
    Point* right_pivot = left_pivot;

    for (Point* curr = left_pivot; curr < endPtr; ++curr) {

        if (crossProduct(c, p2, *curr) > EPSILON) {
            std::iter_swap(curr, right_pivot);
            right_pivot++;
        }
    }

    // Rekurzija za desni trougao
    // Opseg: [left_pivot, right_pivot)
    findHullOptimized(left_pivot, right_pivot, c, p2);
}

void quickHull(vector<Point>& points) {
    // 1. Reset
    hull.clear();
    size_t n = points.size();
    if (n < 3) return;

    // Rezervacija je kljucna da se hull vektor ne bi realocirao
    hull.reserve(n);

    // 2. Nalazenje Min i Max X
    size_t min_idx = 0;
    size_t max_idx = 0;

    // Brzi prolaz
    for (size_t i = 1; i < n; i++) {
        if (points[i].x < points[min_idx].x) min_idx = i;
        if (points[i].x > points[max_idx].x) max_idx = i;
    }

    Point pA = points[min_idx];
    Point pB = points[max_idx];

    hull.push_back(pA);
    hull.push_back(pB);

    // Moramo skloniti pA i pB da nam ne smetaju u petljama
    // Stavljamo pA na indeks 0, pB na indeks 1

    std::swap(points[0], points[min_idx]);
    // Ako je max bio na 0, sada je na min_idx mjestu
    if (max_idx == 0) max_idx = min_idx;
    std::swap(points[1], points[max_idx]);

    // Pointeri za rad
    Point* start = &points[0];
    Point* workerStart = start + 2; // Krecemo od indeksa 2
    Point* end = start + n;

    // 1. Izdvajamo "Gornji" skup (Lijevo od A->B)
    Point* split_point = workerStart;
    for (Point* curr = workerStart; curr < end; ++curr) {
        if (crossProduct(pA, pB, *curr) > EPSILON) {
            std::iter_swap(curr, split_point);
            split_point++;
        }
    }

    // Poziv rekurzije za gornji dio
    findHullOptimized(workerStart, split_point, pA, pB);

    // 2. Izdvajamo "Donji" skup (Desno od A->B, tj. Lijvo od B->A)
    Point* split_point_2 = split_point;
    for (Point* curr = split_point; curr < end; ++curr) {
        // Koristimo < -EPSILON za desnu stranu
        if (crossProduct(pA, pB, *curr) < -EPSILON) {
            std::iter_swap(curr, split_point_2);
            split_point_2++;
        }
    }

    // Poziv rekurzije za donji dio (obrnut redoslijed tacaka linije)
    findHullOptimized(split_point, split_point_2, pB, pA);
}

int main() {
    // --- GENERISANJE PODATAKA ---
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 2.0 * PI);

    int N = 10000;
    double radius = 100000000.0;

    cout << "Generisanje " << N << " tacaka..." << endl;
    vector<Point> points;
    points.reserve(N);

    for (int i = 0; i < N; i++) {
        double angle = dis(gen);
        points.push_back({ radius * cos(angle), radius * sin(angle) });
    }

    clock_t start = clock();
    quickHull(points);
    clock_t end = clock();

    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    cout << "Vrijeme: " << elapsed_secs << " sekundi" << endl;
    cout << "Tacaka u omotacu: " << hull.size() << endl;

    return 0;
}