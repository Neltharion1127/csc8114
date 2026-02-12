# 📄 IEEE Conference Paper Template

This is the LaTeX template for our group assignment paper, based on the standard `IEEEtran` format.

## 🚀 Quick Start

### 1. File Description
- `csc8114.tex`: Main paper file. Write your content here.
- `refs.bib`: Bibliography database. Add your references here.

### 2. How to Write
1. **Update Author Information**:
   - Open `csc8114.tex` and find the `\author{...}` section.
   - Replace names, school, and emails. It is configured for 7 authors sharing the same affiliation.

2. **Write Content**:
   - Find sections like `\section{Introduction}`.
   - Remove `\lipsum[...]` placeholders and insert your actual content.

3. **Add References**:
   - Add BibTeX entries to `refs.bib` (Recommended: Export from Google Scholar).
   - Cite in text using `\cite{key}`, e.g., `\cite{ref1}`.

### 3. How to Compile

#### Method 1: Local Installation (VS Code + LaTeX)
1. **Install LaTeX Distribution**:
   - **Mac**: Install [MacTeX](https://www.tug.org/mactex/) (large download).
   - **Windows**: Install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/).
2. **Setup VS Code**:
   - Install VS Code.
   - Install the **LaTeX Workshop** extension by James Yu.
3. **Compile**:
   - Open the project folder in VS Code.
   - Open `paper/csc8114.tex`.
   - Click the `✅ Build LaTeX project` icon from the left sidebar, or press `Cmd+Option+B` (Mac) / `Ctrl+Alt+B` (Win).
   - The PDF will be generated automatically.

#### Method 2: Online (Overleaf)
1. Go to [Overleaf.com](https://www.overleaf.com) and log in.
2. Click **New Project** -> **Upload Project**.
3. Upload the entire `paper` folder (or zip it first).
4. Set `csc8114.tex` as the main document.
5. Click **Recompile**.
6. **Note**: Overleaf includes all necessary packages automatically.

### 4. Git Workflow (Single Branch - Simple)
For simplicity, we will all work on the `main` branch. **Please follow these rules strictly to avoid conflicts:**

1. **Before you start**: Always `git pull` to get the latest changes.
2. **Edit**: Make your changes.
3. **Commit**:
   ```bash
   git add .
   git commit -m "update section x"
   ```
4. **Push**: `git push`
   - If it fails (someone else pushed), do `git pull` again, fix any conflicts, then `git push`.

**💡 Pro Tip**: Assign each person a specific section (e.g., Alice writes Introduction, Bob writes Methodology). If you only edit your own section, conflicts are very rare!

## ⚠️ Notes
- Build artifacts (`.aux`, `.log`, etc.) are ignored by git.
- Do not modify `IEEEtran.cls` (standard class file).

---


# 📄 IEEE Conference Paper Template（Chinese）

这是小组作业论文的 LaTeX 模版。基于标准的 `IEEEtran` 格式。

## 🚀 快速开始

### 1. 文件说明
- `csc8114.tex`: 主论文文件。在此处编写正文。
- `refs.bib`: 参考文献数据库。在此处添加引用条目。

### 2. 如何编写
1. **修改作者信息**: 
   - 打开 `csc8114.tex`，找到 `\author{...}` 部分。
   - 替换姓名、学院、邮箱即可。已配置为 7 人共用同一机构的格式。
   
2. **编写正文**:
   - 找到 `\section{Introduction}` 等章节。
   - 删除 `\lipsum[...]` 占位符，填入实际内容。
   
3. **添加引用**:
   - 在 `refs.bib` 中添加 BibTeX 条目（推荐从 Google Scholar 导出）。
   - 在文中用 `\cite{key}` 引用，例如 `\cite{ref1}`。

### 3. 如何编译

#### 方法 1: 本地安装 (VS Code + LaTeX)
如果你已经安装了 LaTeX 环境（MacTeX/MiKTeX），推荐使用 VS Code + LaTeX Workshop 插件。
- 直接打开 `paper` 文件夹，保存 `csc8114.tex` 即可自动编译。

#### 方法 2: 在线编译 (Overleaf)
不需要安装任何软件，直接在网页上写。

1. 注册并登录 [Overleaf.com](https://www.overleaf.com)。
2. 点击左上角 **New Project** -> **Upload Project**。
3. 把我们的 `paper` 文件夹直接拖进去（或者打包成 zip 上传）。
4. 点击 **Recompile** 按钮即可看到 PDF。
   - 如果你想复制很多份，可以把项目设为 Private，然后每个人自己建一个 Project 把代码复制过去即可。

### 4. Git 协作教程 (单分支简单版)
为了简单起见，我们统一在 `main` 分支上工作。**请严格遵守以下规则，否则会覆盖别人的代码：**

1. **修改前必做**: `git pull` (拉取最新代码)。
2. **修改代码**: 正常编辑。
3. **提交**:
   ```bash
   git add .
   git commit -m "修改了xx部分"
   ```
4. **推送**: `git push`
   - 如果推送失败（说明有人比你先提交了），请先执行 `git pull`，解决冲突（如果有的话），然后再 `git push`。

**💡 避坑指南**: 大家提前分工好，比如 A 写 Introduction，B 写 Methodology。**只要每个人只改自己负责的那一部分代码，几乎不会发生冲突！**

## ⚠️ 注意事项
- 编译生成的文件（`.aux`, `.log` 等）已被忽略，不会提交到 Git。
- 请勿修改 `IEEEtran.cls` 文件（这是标准类文件）。


