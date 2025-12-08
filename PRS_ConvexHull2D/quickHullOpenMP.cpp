#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <random>    
#include <ctime> 
#include <chrono>
#include <omp.h> 

using namespace std;

const double PI = 4.0 * atan(1.0);
const double EPSILON = 1e-13;

// Dubina rekurzije 3 ce generisati do 8 (2^3) nezavisnih zadataka
const int DECOMPOSITION_DEPTH = 3;

struct Point {
    double x, y;
};

// Struktura koja opisuje jedan posao za thread
struct HullTask {
    Point* startPtr;  // Pokazivac na pocetak niza tacaka za ovaj zadatak
    Point* endPtr;    // Pokazivac na kraj niza
    Point p1;         // Prva tacka linije koja formira bazu trougla
    Point p2;         // Druga tacka linije
};

// Vektorski proizvod (determinanta)
// det > 0 => p je lijevo od AB, det < 0 => p je desno, det = 0 => p je na AB
// inline: Eliminise trosak funkcijskog poziva (function call overhead) za kriticne male operacije koje se izvrsavaju unutar petlji
inline double crossProduct(const Point& a, const Point& b, const Point& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// --- SEKVENCIJALNA VERZIJA (WORKER) ---
// Ovu funkciju izvrsava pojedinacni thread nakon sto preuzme zadatak (Task)
// Radi klasicni rekurzivni QuickHull unutar dodijeljenog opsega memorije
void findHullSeq(Point* startPtr, Point* endPtr, const Point p1, const Point p2, vector<Point>& output) {
    // Bazni slucaj, nema vise tacaka u opsegu
    if (startPtr >= endPtr) return;

    Point* max_ptr = nullptr;
    double max_dist = -1.0;

    // Trazimo najudaljeniju tacku 
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
    output.push_back(c);   // Dodajemo pronadjenu tacku u lokalni rezultat ovog threada

    // Stavljamo najudaljeniju tacku (c) na pocetak opsega i preskacemo je
    std::iter_swap(startPtr, max_ptr);
    Point* workerStart = startPtr + 1;

    // Grupisemo tacke koje su lijevo od p1->c na pocetak niza
    Point* left_pivot = workerStart;
    for (Point* curr = workerStart; curr < endPtr; ++curr) {
        if (crossProduct(p1, c, *curr) > EPSILON) {
            std::iter_swap(curr, left_pivot);
            left_pivot++;
        }
    }

    // Rekurzivni poziv za lijevu stranu (p1 -> c)
    findHullSeq(workerStart, left_pivot, p1, c, output);

    Point* right_pivot = left_pivot;
    for (Point* curr = left_pivot; curr < endPtr; ++curr) {
        if (crossProduct(c, p2, *curr) > EPSILON) {
            std::iter_swap(curr, right_pivot);
            right_pivot++;
        }
    }

    // Rekurzivni poziv za desnu stranu (c -> p2)
    findHullSeq(left_pivot, right_pivot, c, p2, output);
}

// --- DEKOMPOZICIJA (TASK GENERATOR) ---
// Ova funkcija ne rjesava problem do kraja, vec ga dijeli na manje dijelove
// Kada rekurzija dosegne 'depth == DECOMPOSITION_DEPTH', kreira se 'HullTask' 
// i stavlja u vektor zadataka koje ce kasnije obraditi OpenMP threadovi
// 'pre_hull' sluzi da sacuvamo tacke omotaca koje pronadjemo tokom ovog procesa podjele
void decomposeAndCollect(Point* startPtr, Point* endPtr, const Point p1, const Point p2, int depth, vector<HullTask>& tasks, vector<Point>& pre_hull) {
    if (startPtr >= endPtr) return;

    // BAZNI SLUCAJ ZA DEKOMPOZICIJU
    // Ako smo dovoljno duboko, ne idemo dalje rekurzivno, vec pripremamo posao za workere
    if (depth >= DECOMPOSITION_DEPTH) {
        tasks.push_back({ startPtr, endPtr, p1, p2 });
        return;
    }

    // Trazimo najudaljeniju tacku
    Point* max_ptr = nullptr;
    double max_dist = -1.0;

    for (Point* curr = startPtr; curr < endPtr; ++curr) {
        double d = crossProduct(p1, p2, *curr);
        if (d > max_dist) {
            max_dist = d;
            max_ptr = curr;
        }
    }

    if (max_ptr == nullptr || max_dist < EPSILON) return;

    Point c = *max_ptr;
    pre_hull.push_back(c); // Cuvamo tacku odmah jer je dio konveksnog omotaca

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

    // Rezervacija je kljucna da se hull vektor ne bi realocirao
    final_hull.reserve(n);

    // Pronalazenje tacaka sa min i max X koordinatom
    size_t min_idx = 0;
    size_t max_idx = 0;
    for (size_t i = 1; i < n; i++) {
        if (points[i].x < points[min_idx].x) min_idx = i;
        if (points[i].x > points[max_idx].x) max_idx = i;
    }

    Point pA = points[min_idx];
    Point pB = points[max_idx];
    final_hull.push_back(pA);   // Ove dvije tacke su sigurno dio omotaca
    final_hull.push_back(pB);

    // Pomjeranje min i max na prva dva mjesta
    std::swap(points[0], points[min_idx]);
    if (max_idx == 0) max_idx = min_idx;   // Ako je max bio na 0, sada je na min_idx mjestu
    std::swap(points[1], points[max_idx]);

    // Definisemo radni opseg (preskacemo prve dvije tacke)
    Point* start = &points[0];
    Point* workerStart = start + 2;
    Point* end = start + n;

    // Inicijalna podjela na gornji i donji skup tacaka
    // Sve tacke iznad linije pA-pB idu u prvu grupu
    Point* split_point = workerStart;
    for (Point* curr = workerStart; curr < end; ++curr) {
        if (crossProduct(pA, pB, *curr) > EPSILON) {
            std::iter_swap(curr, split_point);
            split_point++;
        }
    }

    // Sve tacke ispod linije pA-pB (odnosno 'iznad' pB-pA) idu u drugu grupu
    Point* split_point_2 = split_point;
    for (Point* curr = split_point; curr < end; ++curr) {
        if (crossProduct(pA, pB, *curr) < -EPSILON) {
            std::iter_swap(curr, split_point_2);
            split_point_2++;
        }
    }

    // --- PRIPREMA ZADATAKA, pravi raspored za threadove
    // Funkcija decomposeAndCollect ce ici par nivoa u dubinu i napuniti vektor tasks
    vector<HullTask> tasks;
    tasks.reserve(64);

    // Obrada gornjeg dijela (iznad pA -> pB)
    decomposeAndCollect(workerStart, split_point, pA, pB, 1, tasks, final_hull);
    // Obrada donjeg dijela (iznad pB -> pA)
    decomposeAndCollect(split_point, split_point_2, pB, pA, 1, tasks, final_hull);

    // --- PARALELNO IZVRSAVANJE (DYNAMIC SCHEDULE) ---
    // Cim thread zavrsi posao, uzima sljedeci, nema cekanja

    int num_tasks = (int)tasks.size();

    // Svaki thread treba svoj privatni vektor za rezultate kako ne bi doslo do race condition kod upisa u zajednicki final_hull
    int max_threads = omp_get_max_threads();
    vector<vector<Point>> thread_results(max_threads);

    // Neki dijelovi omotaca imaju puno tacaka, neki malo
    // Dynamic omogucava da cim thread zavrsi jedan zadatak, uzima sljedeci s liste sto osigurava da svi threadovi rade cijelo vrijeme 
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_tasks; ++i) {
        int tid = omp_get_thread_num();  // ID trenutnog threada
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
    omp_set_num_threads(8);

    // Podesavanje generatora slucajnih brojeva
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 2.0 * PI);

    int N = 100000;
    double radius = 10000000.0;

    cout << "Generisanje " << N << " tacaka" << endl;
    vector<Point> points;
    points.reserve(N);

    // Generisanje tacaka na kruznici (najgori slucaj za Convex Hull)
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
