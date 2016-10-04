
file(GLOB trailsSources *.cpp)
file(GLOB trailsIncludes *.h )

cuda_add_executable(trails ${trailsSources})
target_link_libraries(trails nvcaffe_parser nvinfer jetson-inference)
