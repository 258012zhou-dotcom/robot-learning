# C++ 与 CMake 基础

## 核心关系

机器人项目常用 Python 进行实验、训练和数据分析，使用 C++ 编写驱动、控制和性能敏感模块。

C++ 工作流程：

```text
源代码 → 编译 → 链接 → 可执行程序 → 测试
struct 组合相关数据。
const 表示对象不应被修改。
const T& 以只读引用传递对象，避免不必要的复制。
头文件声明公开接口，源文件提供实现。
非法参数可以抛出 std::invalid_argument。
CMake 与测试
CMake 负责描述构建目标和依赖，实际编译仍由 g++ 完成。
add_library 创建可复用库。
add_executable 创建程序或测试。
target_link_libraries 建立目标之间的依赖。
CTest 根据测试程序的退出码判断成功或失败。
VS Code 和编译器主要发现语法与类型问题；逻辑正确性需要测试验证。