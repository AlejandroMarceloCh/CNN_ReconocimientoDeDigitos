cmake_minimum_required(VERSION 3.10)

# Nombre del proyecto
project(ProyectoProgra3)

# Incluir directorios de cabecera
include_directories(include)

# Archivos fuente
set(SOURCES
    src/main.cpp
    src/mnist_loader.cpp
)

# Dependencias de OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Compilar el ejecutable
add_executable(${PROJECT_NAME} ${SOURCES})

# Enlazar las bibliotecas de OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
