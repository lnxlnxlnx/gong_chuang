下面按**环境管理、包管理、自身配置、实用技巧**四大类，整理 Conda 最常用的命令，附示例和说明。

## 一、环境管理（核心高频）

这是 Conda 最常用的功能，用于创建、切换、管理独立的 Python 环境。

| 功能 | 命令 | 示例/说明 |
| --- | --- | --- |
| 查看所有环境 | `conda env list`    或 `conda info -e` | 列出本机所有 conda 环境，带 `*` 的是当前激活环境 |
| 创建新环境 | `conda create -n 环境名 python=版本号` | `conda create -n torch python=3.10`    创建时可直接带包：`conda create -n test python=3.9 numpy pandas` |
| 激活环境 | `conda activate 环境名` | `conda activate torch`    激活 base 环境：`conda activate base` |
| 退出当前环境 | `conda deactivate` | 退回到上一层环境，连续执行可完全退出 conda |
| 删除环境 | `conda remove -n 环境名 --all`    或 `conda env remove -n 环境名` | `conda remove -n test --all`    必须加 `--all` 才会删除整个环境 |
| 克隆环境 | `conda create -n 新环境名 --clone 旧环境名` | `conda create -n torch_backup --clone torch`    常用于备份或快速复制环境 |
| 重命名环境 | 无直接命令，先克隆再删除旧环境 | `conda create -n new_name --clone old_name`    `conda remove -n old_name --all` |
| 导出环境配置 | `conda env export > environment.yml` | 将当前环境的所有包和版本导出为 yaml 文件，用于分享/复现 |
| 从配置文件创建环境 | `conda env create -f environment.yml` | 根据 yaml 文件一键还原完全一致的环境 |

## 二、包管理

针对**当前激活环境**安装、卸载、更新 Python 包。

| 功能 | 命令 | 示例/说明 |
| --- | --- | --- |
| 查看已安装的包 | `conda list` | 查看指定环境：`conda list -n torch` |
| 安装包 | `conda install 包名` | 指定版本：`conda install numpy=1.24.3`    指定频道（源）：`conda install pytorch -c pytorch`    同时装多个：`conda install pandas matplotlib` |
| 卸载包 | `conda remove 包名` | 指定环境卸载：`conda remove -n torch numpy` |
| 更新包 | `conda update 包名` | 更新当前环境所有包：`conda update --all` |
| 搜索可用包 | `conda search 包名` | `conda search numpy` 查看所有可安装的版本 |

## 三、Conda 自身与配置

### 1. 基础信息

```
conda --version          # 查看 conda 版本
conda info               # 查看 conda 详细信息（路径、源、平台等）
```

### 2. 更新 Conda 自身

```
conda update conda       # 更新 conda 本体（建议在 base 环境执行）
conda update anaconda    # 更新 anaconda 整体套件
```

### 3. 镜像源配置（国内加速必备）

```
# 查看当前配置的源
conda config --show channels

# 添加清华源（常用）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes


# 显示下载时的通道地址（方便排查源问题）
conda config --set show_channel_urls yes

# 恢复默认官方源
conda config --remove-key channels
```

### 4. 清理缓存（释放磁盘空间）

```
conda clean -a           # 清理所有缓存包、索引、临时文件
conda clean -p           # 清理无用的旧版本包
```

## 四、实用技巧

1. **不激活也能操作环境**：大部分命令加 `-n 环境名` 即可直接指定环境，不用先激活```
conda install -n torch numpy
```
2. **conda 与 pip 配合**：conda 找不到的包，激活环境后用 `pip install` 安装，会装到当前 conda 环境里
3. **环境隔离原则**：尽量不在 base 环境装业务包，每个项目单独建环境，避免版本冲突
