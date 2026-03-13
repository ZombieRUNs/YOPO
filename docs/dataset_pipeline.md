# YOPO 数据集构建流程详解

## 总览

YOPO 的数据集构建分为两个阶段：

1. **离线采集**：用 Unity/Flightmare 仿真器随机生成障碍物场景，让无人机在场景中随机采样位姿，采集深度图 + 位姿标签保存到磁盘。
2. **在线训练**：训练时从磁盘加载深度图，通过随机速度/加速度采样做数据增广，C++ 环境实时计算每条候选轨迹的碰撞代价作为监督信号。

---

## 阶段一：离线数据采集（`run/data_collection_simulation.py`）

### 流程

```
Unity 仿真器启动
    ↓
随机生成树木障碍物场景（spawnTreesAndSavePointcloud）
    ↓  ←── 同时保存全局点云为 pointcloud-N.ply
每个场景中随机采样 N 个无人机位姿
    ↓
调用 getDepthImage() 获取该位姿下的深度图
    ↓
保存深度图为 img_K.tif + 位姿标签为 label.npz
    ↓
切换到下一个场景，重复
```

### Flightmare 能返回什么

**全局点云**：通过 `spawnTreesAndSavePointcloud()` 生成场景时，Unity 端会同时渲染全局场景并输出 `.ply` 格式的点云文件，分辨率 0.2m，覆盖范围约 80m × 80m × 11m 的包围盒。这份点云用于后续在 C++ 端构建 ESDF。

**给定位姿的深度图**：通过 `setState(pos, vel, acc, quat)` 将无人机摆放到指定位姿后，调用 `getDepthImage()` 即可获取该视角下的深度图。相机固定安装在机体坐标系下，参数如下：

```yaml
# flightlib/configs/quadrotor_env.yaml
rgb_camera_left:
  t_BC: [0.0, 0.0, 0.1]    # 安装偏移：机体正前方 10cm
  r_BC: [0.0, 0.0, -90]    # 旋转：-90° yaw（朝向飞行方向）
  width: 160
  height: 90
  fov: 90.0                 # 水平视场角
  enable_depth: yes
```

返回的深度图格式：
- 分辨率：160 × 90，float32，单位：米
- Python 侧处理：截断到 20m，归一化到 [0, 1]，NaN 填为 1.0，再 resize 到 160 × 160

```python
# vec_env_wrapper.py
depth = np.clip(depth, 0, 20) / 20.0
depth[np.isnan(depth)] = 1.0
depth = cv2.resize(depth, (160, 160), interpolation=cv2.INTER_CUBIC)
```

### 磁盘存储格式

```
/dataset/
├── 0/                       # 场景 0
│   ├── label.npz            # 所有采样帧的位姿
│   │     positions:   [N, 3]   float32   世界系 xyz
│   │     quaternions: [N, 4]   float32   wxyz 格式
│   ├── img_0.tif            # 第 0 帧深度图（160×90，float32，米）
│   ├── img_1.tif
│   └── ...
├── 1/                       # 场景 1
│   └── ...
└── pointcloud-0.ply         # 场景 0 的全局点云（0.2m 分辨率）
```

---

## 阶段二：训练时的数据使用

### 从磁盘加载

`flightpolicy/yopo/dataloader.py` 的 `YopoDataset.__getitem__` 做了以下处理：

1. 读取 `.tif` 深度图，归一化（同采集阶段的处理）
2. 读取对应帧的 `position` + `quaternion`
3. **随机采样速度和加速度**（不来自仿真，而是按照 `traj_opt.yaml` 中的分布随机生成）：
   - 前向速度 $v_x$：对数正态分布，均值 1.5，方差 0.15（单位：$v_{max}=6$ m/s 归一化）
   - 侧向速度 $v_y, v_z$：正态分布，均值 0，方差 0.45 / 0.10
   - 加速度：正态分布，小方差（0.028 / 0.05）
4. 随机生成目标方向（正前方 + 小扰动，体坐标系下）

这种设计使得一张深度图可以在不同的速度/加速度条件下被复用，**大幅提升了数据利用率**，不需要为每个速度状态重新采集深度图。

### 训练样本结构

网络每次看到的输入是：

| 张量 | 形状 | 含义 |
|------|------|------|
| `depth` | `[B, 1, 160, 160]` | 归一化深度图 |
| `obs_input` | `[B, 9, 3, 5]` | 速度 + 加速度 + 目标方向，在 15 个 primitive 坐标系下各旋转一次，铺成 3×5 网格 |

`obs_input` 的 9 个通道 = `[v_x, v_y, v_z, a_x, a_y, a_z, g_x, g_y, g_z]`，对于网格中每个 primitive 位置 `(i, j)`，都把这 9 个量从机体系旋转到该 primitive 的局部坐标系。

---

## ESDF 构建（C++ 侧，训练时）

训练时 C++ 环境在初始化时加载 `.ply` 点云，实时构建每个场景的 ESDF（Euclidean Signed Distance Field）：

```
pointcloud-N.ply
    ↓
TrajOptimizationBridge::SdfConstruction()
    ↓ 使用 sdf_tools 库
SignedDistanceField (体素地图，0.2m 分辨率)
    ↓ 缓存在 esdf_maps[N] 中
训练时按 map_id 索引取用
```

训练批次中每条样本都携带 `map_id`，调用 `env.setMapID(map_id)` 后，后续 `getCostAndGradient()` 就会用对应场景的 ESDF 计算碰撞代价。

---

## 监督信号来源：`CostAndGradLayer`

这是整个训练流程最关键的部分，也是 YOPO 不依赖人工标注的原因。

```
网络预测 15 条轨迹的终态 endstate_pred [B, 15, 9]
    ↓ 每个终态是 (位置, 速度, 加速度) 各 3 维
env.getCostAndGradient(endstate_pred, traj_id)
    ↓ C++ 端：用 5 阶多项式拟合从当前状态到预测终态的轨迹
    ↓         沿轨迹查询 ESDF 累积碰撞代价
    ↓         对终态的 9 维向量求梯度（数值微分）
返回 cost_label [B, 15]，grad [B, 15, 9]
    ↓
cost_label 作为 score 分支的回归目标
grad 通过 CostAndGradLayer 的自定义 backward 传回 endstate 分支
```

```python
# yopo_network.py - 自定义 autograd Function
class CostAndGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, endstate, cost, grad):
        ctx.save_for_backward(grad)
        return cost                        # 直接返回 C++ 算好的代价

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output, None, None   # 用 C++ 梯度反传
```

这样，整个轨迹优化问题的梯度就通过这个"桥梁"传到了神经网络权重上，无需任何人工标注轨迹。

---

## 代价函数构成（`traj_opt.yaml`）

C++ 端计算的轨迹代价由以下几项加权求和：

| 项 | 权重参数 | 含义 |
|----|----------|------|
| 平滑代价 | `ws = 4e-5` | 轨迹加加速度（jerk）积分 |
| 碰撞代价 | `wc = 1e-3` | 沿轨迹对 ESDF 取负距离累积 |
| 目标代价 | `wg = 2e-4` | 终态与目标方向的偏差 |
| 轨迹长度 | `wl` | 总路径长度惩罚 |

规划水平：4m（`radio_range: 4.0`，实际单侧范围），FOV 5×3 网格覆盖水平 90°、垂直 60°。

---

## Python ↔ C++ 接口汇总（PyBind11）

关键接口定义在 `flightlib/src/wrapper/pybind_wrapper.cpp`：

```python
env = flightgym.QuadrotorEnv_v1(cfg_path, render)

# 场景初始化
env.spawnTreesAndSavePointcloud(num_trees, save_path)   # 生成场景 + 保存点云
env.setMapID(map_id)                                     # 切换场景（加载对应 ESDF）

# 状态设置
env.setState(pos, vel, acc, quat)                        # 设置无人机状态
env.setGoal(goal_pos)                                    # 设置目标位置

# 传感器数据
env.getDepthImage(depth_buffer)                         # 获取 160×90 深度图
env.getObs(obs_buffer)                                  # 获取 13 维状态向量

# 训练核心
env.getCostAndGradient(endstate, traj_id, cost, grad)   # 轨迹代价 + 梯度
```

`getObs` 返回的 13 维向量：`[pos(3), vel(3), acc(3), quat(4)]`，其中 vel/acc 在**机体坐标系**，pos/quat 在**世界坐标系**。

---

## 小结

| 环节 | 来源 | 格式 |
|------|------|------|
| 深度图 | Flightmare 渲染（给定位姿） | 160×90 float32，归一化到 [0,1] |
| 全局点云 | Flightmare 场景生成时导出 | `.ply`，0.2m 分辨率 |
| ESDF | C++ 从点云实时构建 | 体素地图，`sdf_tools` |
| 速度/加速度 | 训练时随机采样（非仿真） | 按 `traj_opt.yaml` 分布生成 |
| 轨迹代价/梯度 | C++ 轨迹优化（ESDF 查询） | `CostAndGradLayer` 桥接到 PyTorch |
| 人工标注 | **无** | 全程自监督 |
