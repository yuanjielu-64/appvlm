# 如何使用异步线程池处理慢Callback

## 概述

`AsyncTaskExecutor` 线程池允许你将耗时的callback任务放到后台执行，避免阻塞其他快速callback。

## 使用场景

- ✅ **视觉识别** (YOLO, 目标检测) - 耗时 50-200ms
- ✅ **全局路径规划** (A*, Dijkstra) - 耗时 20-100ms
- ✅ **地图处理** (大规模costmap更新) - 耗时 > 10ms
- ❌ **快速callback** (odometry, laser) - 耗时 < 1ms，直接执行即可

---

## 示例1: 视觉识别Callback（最常见）

```cpp
// Jackal_callbacks.cpp

void JackalCallbacks::visionCallback(const sensor_msgs::Image::ConstPtr& msg) {
    // 将耗时的视觉处理提交到线程池
    robot_->async_executor_->submit([this, msg]() {
        // === 这段代码在后台线程执行 ===

        // 1. 图像预处理 (20ms)
        cv::Mat image = convertRosImageToCvMat(msg);

        // 2. YOLO检测 (100ms)
        auto detections = runYOLODetection(image);

        // 3. 结果写回需要加锁（因为主线程可能读取）
        {
            std::lock_guard<std::mutex> lock(robot_->vision_mutex_);
            robot_->detected_objects = detections;
        }

        ROS_INFO("Vision processing completed: %zu objects detected", detections.size());
    });

    // callback立即返回，不阻塞其他callback
}
```

**需要添加的代码**：

1. 在 `Jackal.hpp` 的 `protected` section 添加mutex：
```cpp
protected:
    std::mutex vision_mutex_;  // 保护视觉识别结果
```

2. 在 `Jackal.hpp` 的 `public` section 添加结果变量：
```cpp
public:
    std::vector<DetectedObject> detected_objects;  // 视觉识别结果
```

3. 在 `Jackal_callbacks.hpp` 添加callback声明：
```cpp
void visionCallback(const sensor_msgs::Image::ConstPtr& msg);
```

4. 在 `Jackal.cpp` 构造函数添加订阅：
```cpp
vision_sub = nh.subscribe("/camera/image", 1, &Robot_config::visionCallback, this);
```

---

## 示例2: 自定义全局路径规划

```cpp
// Jackal_callbacks.cpp

void JackalCallbacks::customGlobalPlanCallback(const geometry_msgs::PoseStamped::ConstPtr& goal) {
    robot_->async_executor_->submit([this, goal]() {
        // === 后台执行耗时的A*算法 ===

        auto start_pose = robot_->getPoseState();
        auto costmap = robot_->getCostMap();

        // 运行A* (50ms)
        std::vector<geometry_msgs::PoseStamped> path =
            computeAStarPath(start_pose, goal, costmap);

        // 写回结果（加锁）
        {
            std::lock_guard<std::mutex> lock(robot_->goal_mutex_);
            robot_->global_path_custom = path;
        }

        ROS_INFO("Custom global path computed with %zu waypoints", path.size());
    });
}
```

---

## 示例3: 并发的Costmap处理（保留原callback逻辑）

如果你想让**现有的globalPathCallback**也异步执行（因为它很复杂）：

```cpp
// Jackal_callbacks.cpp

void JackalCallbacks::globalPathCallback(const nav_msgs::Path::ConstPtr& msg) {
    // 检查是否需要异步处理
    if (msg->poses.size() > 100) {  // 路径很长时，异步处理
        robot_->async_executor_->submit([this, msg]() {
            // 后台执行原来的复杂逻辑
            processGlobalPath(msg);
        });
    } else {  // 路径短，直接处理
        processGlobalPath(msg);
    }
}

// 将原来的处理逻辑抽取到独立函数
void JackalCallbacks::processGlobalPath(const nav_msgs::Path::ConstPtr& msg) {
    // ... 原来globalPathCallback中的所有代码 ...

    // 如果修改共享数据，需要加锁
    {
        std::lock_guard<std::mutex> lock(robot_->goal_mutex_);
        robot_->local_goal = lg;
        robot_->local_paths = paths;
    }
}
```

---

## 线程安全检查清单

使用异步callback时，遵循这些规则：

### ✅ 安全操作（不需要加锁）
```cpp
// 1. 只读取msg数据
auto image = msg->data;

// 2. 创建局部变量
std::vector<double> local_result;

// 3. 调用纯函数
auto output = processPureFunction(input);
```

### ⚠️ 需要加锁的操作
```cpp
// 1. 写入robot_的成员变量
{
    std::lock_guard<std::mutex> lock(robot_->data_mutex_);
    robot_->detected_objects = results;
}

// 2. 读取robot_的成员变量（如果其他callback可能修改）
{
    std::lock_guard<std::mutex> lock(robot_->state_mutex_);
    double x = robot_->robot_state.x_;
}
```

### ❌ 危险操作（绝对避免）
```cpp
// 1. 不加锁读写共享数据
robot_->detected_objects = results;  // ← 数据竞争！

// 2. 在锁内执行耗时操作
{
    std::lock_guard<std::mutex> lock(robot_->mutex_);
    runYOLO(image);  // ← 阻塞其他线程100ms！
}
```

---

## 性能对比

### 不使用线程池（当前）
```
时间轴 →
0ms:  visionCallback开始执行
120ms: visionCallback结束
120ms: laserCallback执行 (被阻塞了120ms!)
```

### 使用线程池（推荐）
```
时间轴 →
0ms:  visionCallback提交任务，立即返回
0.1ms: laserCallback执行 ✓ (不被阻塞)
0.2ms: odometryCallback执行 ✓

后台线程:
0ms~120ms: 并行执行视觉识别
```

---

## 调试技巧

### 1. 查看线程池状态
```cpp
ROS_INFO("Pending async tasks: %zu", robot_->async_executor_->pendingTasks());
```

### 2. 测量任务执行时间
```cpp
robot_->async_executor_->submit([this, msg]() {
    auto start = std::chrono::steady_clock::now();

    // 你的耗时任务
    processVision(msg);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    ROS_INFO("Vision task took %ld ms", duration.count());
});
```

---

## 总结

**何时使用异步线程池**：
- 任务耗时 > 10ms
- 不想阻塞其他callback
- 可以容忍轻微的结果延迟

**何时直接执行**：
- 任务很快 (< 1ms)
- 需要立即得到结果
- 逻辑简单，不涉及复杂计算
