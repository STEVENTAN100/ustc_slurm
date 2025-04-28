### 科大超算集群实验常用命令

#### 基本Linux命令学习

`faq` 命令可以快捷查询常用Linux命令

#### tmux常用命令

新建tmux终端：
```sh
tmux new -s <session-name>
```

tmux复制模式：先按`Ctrl+b`，再按下`[`，按`q`退出复制模式

tmux会话分离detach（即让终端后台运行）：先按`Ctrl+b`，再按下`d`

tmux查看已有终端：
```sh
tmux ls
```

tmux回到原先终端：
```sh
tmux attach -t <session-name>
```

tmux结束会话：进入到原先终端，输入`exit`

效果演示：
```text
zhangxuling@hanhai22-01$ tmux attach -t cv
[detached (from session cv)]
zhangxuling@hanhai22-01:$ tmux ls
cv: 1 windows (created Mon Apr 21 10:00:43 2025)
zhangxuling@hanhai22-01$ tmux attach -t cv
[exited]
```

#### `srun`跑实验命令，建议配合tmux使用，和`sbatch`一个效果

<a id="srun"></a>
基本格式：
```sh
srun -p GPU-8A100 -N 1 -c 8 -n 1 --gres=gpu:1 --qos=gpu_8a100 python -u emnist.py
```

参数说明：
```
-p 指定作业提交的队列，无须修改。
-N 指定作业申请节点数，一般都是1，一般不修改。
-c 指定作业申请CPU核数，一般8核或16核
-n 指定作业申请进程数，一般都是1，无需修改。
--gres= 指定需要的资源，例如需要一个gpu: --gres=gpu:2,如作业是跨节点的,则表示每个节点申请2个gpu
--time= 指定作业运行的最长时间，时间<time>的格式可以为：分钟、分钟:秒、小时:分钟:秒、天-小时、天-小时:分钟、天-小时:分钟:秒
--qos= 指定作业使用的qos，无需修改。
```
后面跟的命令可以是`python`或`bash`命令，[实验形式](#实验形式)那一节会说明

#### `sbatch`跑实验命令，特殊情况需要

首先写好`gpu_job.sh`（更多参数详见[这里](https://scc.ustc.edu.cn/zlsc/user_doc/html/slurm/slurm.html#id25)或`faq`）：
```sh
#!/bin/bash
#SBATCH -J test_pytorch
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1 -c 4 -n 2
#SBATCH --gres=gpu:2
#SBATCH --qos=gpu_8a100

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu core.

# 加载系统模块 (不是必须)
module purge
module load cuda/12.4

# 检查环境是否正确加载
echo "Python path: $(which python)"
echo "PyTorch version info:"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

python -u emnist.py
```
然后执行：
```sh
sbatch gpu_job.sh
```

我个人觉得这种方式比较麻烦，没有特殊情况使用上小节的命令足够

#### 资源申请迟迟等不到分配

查看节点在线情况：`sinfo`，只有`mix`状态的节点我们才能用，`alloc`是管理员才有的命令
```text
PARTITION     AVAIL  TIMELIMIT  NODES  STATE NODELIST
GPU-8A100        up 5-00:00:00      1   plnd gnode02
GPU-8A100        up 5-00:00:00     13    mix gnode[01,03,05,07-08,11-15,18-20]
GPU-8A100        up 5-00:00:00      1  alloc gnode06
```

查看排队情况：`squeue | grep GPU`
```text
 440097 GPU-8A100      413     hzhh PD       0:00      1 (Resources)
446884 GPU-8A100 GRPO_bac jiwc0606 PD       0:00      2 (Resources)
447846 GPU-8A100 fair-che   yinshi PD       0:00      1 (Priority)
448492 GPU-8A100 job_name solomonz PD       0:00      3 (Priority)
448923 GPU-8A100 GRPO_bac jiwc0606 PD       0:00      1 (Priority)
448986 GPU-8A100    webrl  swtuser PD       0:00      1 (Priority)
440096 GPU-8A100      413     hzhh  R 2-03:30:42      1 gnode13
443229 GPU-8A100    28-54 yangjuny  R 4-23:24:05      1 gnode20
```

查看某个节点gnode03分配的情况：`scontrol show node gnode03`
```text
NodeName=gnode03 Arch=x86_64 CoresPerSocket=32 
   CPUAlloc=48 CPUEfctv=64 CPUTot=64 CPULoad=40.67
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:a100:8
   NodeAddr=gnode03 NodeHostName=gnode03 Version=22.05.8
   OS=Linux 5.15.0-60-generic #66-Ubuntu SMP Fri Jan 20 14:29:49 UTC 2023 
   RealMemory=1000000 AllocMem=0 FreeMem=967475 Sockets=2 Boards=1
   State=MIXED ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=GPU-8A100 
   BootTime=2025-04-08T09:07:23 SlurmdStartTime=2025-04-08T09:44:26
   LastBusyTime=2025-04-20T15:53:48
   CfgTRES=cpu=64,mem=1000000M,billing=64,gres/gpu=8
   AllocTRES=cpu=48,gres/gpu=8
```
只需关注`Alloc`的部分，CPU分配小于64核，gpu分配小于8表示有空闲资源

<a id="实验形式"></a>
#### 实验形式

这里回顾[`srun`基本格式](#srun)的后一部分，跑`py`脚本的代码不再赘述，`-u`参数是为了能实时打印出来运行的结果进行调试，当完成调试真正跑实验时不需要加

还有另外一种形式的实验使用`sh`脚本，适用于python命令后面有超多参数的情况，运行命令举例：`bash test.sh`

特殊情况下，比如说画图或某些变态实验要求，必须使用`Jupyter Notebook`的情况下，在`pip`安装好`jupyter`的前提下，运行命令为：
```sh
jupyter nbconvert --execute --to notebook --inplace <notebook>.ipynb
```

我倾向于原地跑实验，所以加了`inplace`参数，当然如果你想了解[更多](https://stackoverflow.com/questions/35545402/how-to-run-an-ipynb-jupyter-notebook-from-terminal)

一些小福利有关`py`转`ipynb`的说明（假设`emnist.py`文件已经符合转`jupyter`的格式，可以让大模型生成模板给你）：
```sh
jupytext --to notebook emnist.py
```

#### 写在最后

谢谢你看到了最后，相信你已经掌握了不少新知识，祝你跑实验发论文顺利！
