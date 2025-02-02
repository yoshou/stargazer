cmake_minimum_required(VERSION 3.18)

find_program(_PROTOBUF_PROTOC protoc PATHS "${VCPKG_ROOT}/installed/x64-windows/tools/protobuf")
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin PATHS "${VCPKG_ROOT}/installed/x64-windows/tools/grpc")

find_package(Protobuf CONFIG REQUIRED PATHS "${VCPKG_ROOT}/installed/x64-windows")
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)

find_package(gRPC CONFIG REQUIRED PATHS "${VCPKG_ROOT}/installed/x64-windows")
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_REFLECTION gRPC::grpc++_reflection)

# Proto file
get_filename_component(sensor_proto "${CMAKE_CURRENT_LIST_DIR}/sensor.proto" ABSOLUTE)
get_filename_component(sensor_proto_path "${sensor_proto}" PATH)

# Generated sources
set(sensor_proto_srcs "${CMAKE_BINARY_DIR}/protos/sensor.pb.cc")
set(sensor_proto_hdrs "${CMAKE_BINARY_DIR}/protos/sensor.pb.h")
set(sensor_grpc_srcs "${CMAKE_BINARY_DIR}/protos/sensor.grpc.pb.cc")
set(sensor_grpc_hdrs "${CMAKE_BINARY_DIR}/protos/sensor.grpc.pb.h")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/protos")

add_custom_command(
      OUTPUT "${sensor_proto_srcs}" "${sensor_proto_hdrs}" "${sensor_grpc_srcs}" "${sensor_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_BINARY_DIR}/protos"
        --cpp_out "${CMAKE_BINARY_DIR}/protos"
        -I "${sensor_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${sensor_proto}"
      DEPENDS "${sensor_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_BINARY_DIR}/protos")

# sensor_grpc_proto
add_library(stargazer_sensor_grpc_proto
  ${sensor_grpc_srcs}
  ${sensor_grpc_hdrs}
  ${sensor_proto_srcs}
  ${sensor_proto_hdrs})
target_link_libraries(stargazer_sensor_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})
