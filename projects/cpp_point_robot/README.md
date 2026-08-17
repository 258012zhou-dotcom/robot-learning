# C++ Point Robot

使用 C++17 实现二维点机器人单步运动，并通过 CMake 和 CTest 构建、测试。

## 构建

```bash
cmake -S projects/cpp_point_robot -B projects/cpp_point_robot/build
cmake --build projects/cpp_point_robot/build
```

## 测试

```bash
ctest --test-dir projects/cpp_point_robot/build --output-on-failure
```

## 运行

```bash
./projects/cpp_point_robot/build/point_robot
```

预期输出：`Next position: (1, 0.5)`。