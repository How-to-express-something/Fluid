# 架构与 CPU/GPU 数据流

## 模块与所有权

- `Application`：输入、固定模拟步长、面板状态、暂停与重置；组织求解与显示。
- `FluidSim`：拥有 CUDA 场缓冲、压力 Graph 和 CUDA 图形注册句柄；借用 GL 纹理。
- `Renderer`：拥有 GL 纹理、着色器、顶点缓冲和 VAO。
- `ControlPanel`：拥有 ImGui context 与后端，借用 GLFW window；不直接拥有或求解流体场。

GL context 必须先创建并 current，再选择对应 CUDA 设备。退出时在 context current 的主线程关闭 UI/回调，注销 CUDA 图形资源，删除 GL 资源，最后销毁窗口。

## 数据在什么位置

| 数据 | CPU 上保存的内容 | 实际数据及访问 |
| --- | --- | --- |
| 参数、输入队列 | 普通 C++ 结构体/容器 | CPU 可直接读写；启动 kernel 时按值传参 |
| `float2* velA/velB` | 设备地址 | cudaMalloc 分配的 GPU 线性内存；CPU 不可直接解引用 |
| 密度、压力、散度指针 | 设备地址 | GPU 线性内存，由核函数读写 |
| GL 纹理编号 | CPU 整数句柄 | 图像内容位于 GPU，由 GL 采样 |
| `cudaGraphicsResource*` | CUDA 图形注册句柄 | 表示共享图形资源，不是密度数组 |
| `cudaArray_t` | 本次映射得到的不透明句柄 | 映射后的 GL 图像视图，不是另分配的一份密度 |

初始化时 CPU 生成噪声数组并执行一次 CPU→GPU 上传。正常步进只更新 GPU 缓冲；发布为 GPU→GPU 复制。只有诊断/测试回读和截图进行 GPU→CPU 传输。

```text
CPU 输入/参数 → 启动 CUDA 工作
                        ↓
GPU velA/B、denA/B、pA/B、div
                        ↓ denA 最新密度
Map GL 纹理 → 获取 cudaArray_t → cudaMemcpy2DToArrayAsync → Unmap
                        ↓
OpenGL 片元着色器采样密度 → framebuffer
```

速度 `float2*` 只在求解器内使用，不直接发布为图像。密度是单通道 float，复制到 R32F 纹理；不能把线性设备指针强转成 `cudaArray_t`。Map 期间 GL 不访问该纹理，Unmap 后才绘制。这条链路包含 GPU 内复制，不称为零复制。

## 每步求解与 Graph

```text
输入/浮力 → 速度平流 → 可选扩散 → 散度
         → 清零压力 → 重放压力 Graph → 投影 → 密度平流
```

场采用紧凑 MAC 网格：`float2.x` 为右侧面速度，`.y` 为下侧面速度；压力/密度在格心。速度和密度的 ping-pong 在 CPU 交换地址，不复制整个数组。压力 Graph 捕获时交换局部输入/输出指针，保存最终压力地址；成员缓冲地址保持固定。奇数次结果在 pB，偶数次在 pA。Graph 重放不会执行 CPU 的 `std::swap`。

压力包装函数显式传入捕获 stream。实例化完成后复用 executable，并在流 0 启动，使散度、清零、求解、投影保持顺序。修改迭代次数后下次 step 重建；resize 和移动同时处理缓冲与 Graph 所有权，释放前等待相关工作完成。

## 输入与固定时间步

渲染频率与模拟频率解耦，累积墙钟时间后以 1/60 秒推进，限制每轮补算次数。暂停/最小化清理积累，避免恢复后无限追赶。

GLFW 后端串联应用回调，UI 始终收到事件；应用按捕获标志执行自身行为。鼠标手势经过 UI 或失焦后需释放再重新按下，避免拖动滑块离开面板时误注入；新手势和窗口几何变化清掉旧位置，避免错误速度。

参数面板只编辑 CPU 设置。Application 在模拟前应用，重置由主循环执行；暂停不因为编辑参数而推进。显示增益/颜色仅影响绘制，不修改场或请求额外密度发布。

## 边界

半拉格朗日平流有数值耗散，密度不严格守恒；固定次数 Jacobi 对大网格低频误差收敛有限。现有检查包含投影、参考扩散、非零输入步进、奇偶迭代、移动/resize、GL 像素与 UI 行为，但没有穷尽全部规模、硬件和操作序列。
