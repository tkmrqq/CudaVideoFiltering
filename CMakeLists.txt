cmake_minimum_required(VERSION 3.27)
project(VideoFiltering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ---- Пути ----
set(FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/src/ffmpeg")
set(NVCODEC_ROOT "${CMAKE_SOURCE_DIR}/src/Video_Codec_SDK_13.0.19")

# ---- FFmpeg headers/libs ----
find_path(AVCODEC_INCLUDE_DIR NAMES libavcodec/avcodec.h PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH)
find_path(AVFORMAT_INCLUDE_DIR NAMES libavformat/avformat.h PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH)
find_path(AVUTIL_INCLUDE_DIR NAMES libavutil/avutil.h PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH)
find_path(SWSCALE_INCLUDE_DIR NAMES libswscale/swscale.h PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH)

find_library(AVCODEC_LIBRARY NAMES avcodec PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
find_library(AVFORMAT_LIBRARY NAMES avformat PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
find_library(AVUTIL_LIBRARY NAMES avutil PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
find_library(SWSCALE_LIBRARY NAMES swscale PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)

# ---- NVIDIA SDK paths ----
set(NVSDK_INCLUDE "${NVCODEC_ROOT}/Interface")
if(EXISTS "${NVCODEC_ROOT}/Lib/x64")
    set(NVSDK_LIB_DIR "${NVCODEC_ROOT}/Lib/x64")
elseif(EXISTS "${NVCODEC_ROOT}/lib/x64")
    set(NVSDK_LIB_DIR "${NVCODEC_ROOT}/lib/x64")
else()
    message(FATAL_ERROR "Cannot find NVIDIA lib directory in ${NVCODEC_ROOT}")
endif()

link_directories("${NVSDK_LIB_DIR}")

message(STATUS "NVIDIA SDK lib: ${NVSDK_LIB_DIR}")

# ---- Основной исполняемый файл ----
add_executable(VideoFiltering src/main1.cpp src/prewitt_cuda.cu src/ImageProcessing.cpp src/MkvMuxer.cpp src/NVEncoder.cpp src/MP4Muxer.cpp)

# ---- Подключаем include пути ----
target_include_directories(VideoFiltering PRIVATE
    ${AVCODEC_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
    ${AVUTIL_INCLUDE_DIR}
    ${SWSCALE_INCLUDE_DIR}
    ${NVSDK_INCLUDE}
)

# ---- CUDA Toolkit ----
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
    # Линкуем CUDA Runtime и CUDA Driver API
    target_link_libraries(VideoFiltering PRIVATE 
        CUDA::cudart
        CUDA::cuda_driver  # <-- ЭТО КЛЮЧЕВОЕ ИЗМЕНЕНИЕ!
    )
endif()

# ---- Линкуем FFmpeg ----
target_link_libraries(VideoFiltering PRIVATE
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
)

# ---- Линкуем NVIDIA SDK ----
target_link_libraries(VideoFiltering PRIVATE 
    nvcuvid 
    nvEncodeAPI
)

# ---- Windows system libs ----
if(WIN32)
    target_link_libraries(VideoFiltering PRIVATE
        ws2_32
        bcrypt
        secur32
        shlwapi
    )
endif()

# ---- Defines ----
target_compile_definitions(VideoFiltering PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    IMAGES_DIR="${CMAKE_SOURCE_DIR}/images"
    VIDEO_DIR="${CMAKE_SOURCE_DIR}/videos"
    OUTPUT_DIR="${CMAKE_SOURCE_DIR}/vout_frames/"
)

message(STATUS "Configuration completed successfully.")
