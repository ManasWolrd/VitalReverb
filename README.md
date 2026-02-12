# plugin template
或许是吧, visual studio/vs code/clion直接打开这个文件夹等cmake配置就行了.  

推荐使用Ninja和clang/GCC编译器, 优化比MSVC好很多.  

# 克隆到本地
```bash
git clone --recurse-submodules https://github.com/ManasWorld/plugin-template.git
```
或者
```bash
git clone https://github.com/ManasWorld/plugin-template.git
cd l-model-plugin-template
git submodule update --init --recursive
```

## MacOS授权

```bash
sudo xattr -dr com.apple.quarantine /path/to/your/plugins/plugin_name.component
sudo xattr -dr com.apple.quarantine /path/to/your/plugins/plugin_name.vst3
sudo xattr -dr com.apple.quarantine /path/to/your/plugins/plugin_name.lv2
```
## 构建

```bash
git clone --recurse-submodules https://github.com/ManasWorld/plugin-template.git

# windows
cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release -S . -B ./build

# linux
sudo apt update
sudo apt-get install libx11-dev libfreetype-dev libfontconfig1-dev libasound2-dev libxrandr-dev libxinerama-dev libxcursor-dev
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -S . -B ./build
cmake --build ./build --config Release

# macOS
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -S . -B ./build
cmake --build ./build --config Release
```
