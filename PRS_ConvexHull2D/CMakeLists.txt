cmake_minimum_required(VERSION 3.10)

# Ime projekta
project(PRS_ConvexHull2D)

# Koristimo C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Uključi podršku za OpenMP (za ompSIMD.cpp i sl.)
find_package(OpenMP REQUIRED)

# Lista fajlova (svaki je poseban program)
set(SOURCES
    main_seq.cpp
    main_seq_circle.cpp
    main_seq_examples.cpp
    ompSIMD.cpp
    quickHullOpenMP.cpp
    quickHullSIMD.cpp
)

# Petlja koja pravi poseban .exe za svaki fajl
foreach(SRC_FILE ${SOURCES})
    
    # Izdvaja ime bez ekstenzije (npr. 'ompSIMD')
    get_filename_component(TARGET_NAME ${SRC_FILE} NAME_WE)

    # Pravi .exe fajl
    add_executable(${TARGET_NAME} ${SRC_FILE})

    # Povezuje OpenMP
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Optimizacije za brzinu i SIMD (AVX2)
    # Optimizacije (SIMD/AVX2)
    if(MSVC)
        # Uvek uključi AVX2 instrukcije
        target_compile_options(${TARGET_NAME} PRIVATE /arch:AVX2)
        
        # Uključi /O2 (brzinu) SAMO ako je konfiguracija 'Release'
        # Ovo rešava konflikt sa /RTC1 u Debug modu
        target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -O3 -mavx2)
    endif()

endforeach()