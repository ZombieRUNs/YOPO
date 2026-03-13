# GCOPTER Replay Notes

该文件保留运行目录相关的最小说明，完整流程文档已统一到仓库根目录：

- `fm-gcopter-dataset/README.md`

---

## Quick Start (from YOPO/run)

### 1) 仅导出深度视频

```bash
cd /root/workspace/YOPO/run
python replay_gcopter_traj.py \
  --dataset_root dataset/gcopter_trajs \
  --scene_id 0 --traj_id 3 \
  --video_out /tmp/scene000_traj0003_depth.mp4 \
  --video_only
```

### 2) ROS + RViz 回放（推荐）

```bash
source /opt/ros/noetic/setup.bash
source /root/workspace/ROS/devel/setup.bash

roslaunch gcopter collect_replay.launch \
  dataset_root:=/root/workspace/YOPO/run/dataset/gcopter_trajs \
  scene_id:=0 traj_id:=3 rate:=30
```

更多参数说明（碰撞检测、RViz 配置、Docker 使用、编译流程）请看根目录 `README.md`。
