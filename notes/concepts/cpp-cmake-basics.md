# C++ 与 CMake 基础

## 快速复习

```text
头文件声明接口 + 源文件实现逻辑
             ↓
CMake 描述库、程序和测试之间的依赖
             ↓
编译、链接 → 可执行程序 / CTest 测试
```

- `struct Vec2` 把 x、y 两个相关数据组合为二维向量。
- `const Vec2&` 表示只读引用：不复制对象，也不应修改它。
- 非法参数用 `std::invalid_argument` 明确报告。
- CMake 的 `add_library` 建立可复用库，`add_executable` 建立程序或测试，`target_link_libraries` 连接依赖。

## 项目中的用法

`projects/cpp_point_robot/` 将 `next_position` 声明在 `include/point_robot.hpp`，实现在 `src/point_robot.cpp`；主程序和测试都链接 `point_robot_lib`。CTest 依据测试程序的退出码判断成功。

Python 目前适合做实验、数据和可视化；C++ 更适合后续的性能敏感模块、驱动和实时控制接口。编译成功只说明代码能构建，逻辑仍需测试验证。
