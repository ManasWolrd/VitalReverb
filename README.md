# VitalReverb
一个vital合成器上混响的移植  
A port of the Reverb plugin from Vital synthesizer.

## Install Plugin / 安装插件

GitHub Release 里的 zip 是按平台分别打包的，解压后目录会多一层平台文件夹，这个是正常的。  
GitHub Release zip files are packaged per platform, so it is normal to see one extra platform folder after extraction.

你真正需要复制的是插件 bundle 本身，不是 `Release` 或 `VST3` / `AU` / `LV2` 这一层目录。  
What you actually need to copy is the plugin bundle itself, not the `Release` or `VST3` / `AU` / `LV2` folder that contains it.

解压后大致会看到这样的结构：  
After extraction, the folder structure will look roughly like this:

```text
pluginName-win-vX.Y.Z.zip
  VST3/
    PluginName.vst3/

pluginName-macos-vX.Y.Z.zip
  AU/
    PluginName.component/
  VST3/
    PluginName.vst3/

pluginName-linux-vX.Y.Z.zip
  LV2/
    PluginName.lv2/
  VST3/
    PluginName.vst3/
```

安装时请直接复制这些文件夹之一：  
When installing, copy one of these folders directly:

- `PluginName.vst3`
- `PluginName.component`
- `PluginName.lv2`

常见安装目录：  
Common install locations:

- Windows VST3: `C:\Program Files\Common Files\VST3\`
- macOS VST3: `/Library/Audio/Plug-Ins/VST3/` or `~/Library/Audio/Plug-Ins/VST3/`
- macOS AU: `/Library/Audio/Plug-Ins/Components/` or `~/Library/Audio/Plug-Ins/Components/`
- Linux LV2: `~/.lv2/` or `/usr/lib/lv2/`
- Linux VST3: `~/.vst3/` or `/usr/lib/vst3/`

例如在 Windows 上，不要复制 `VST3` 文件夹本身，而是把其中的 `PluginName.vst3` 整个文件夹复制到 `C:\Program Files\Common Files\VST3\`。  
For example, on Windows, do not copy the `VST3` folder itself. Copy the whole `PluginName.vst3` folder inside it to `C:\Program Files\Common Files\VST3\`.

额外的，macOS 用户还需要做以下工作：  
Additionally, macOS users may need to do the following:

```bash
sudo xattr -dr com.apple.quarantine /path/to/your/plugins/plugin_name.component
sudo xattr -dr com.apple.quarantine /path/to/your/plugins/plugin_name.vst3
sudo xattr -dr com.apple.quarantine /path/to/your/plugins/plugin_name.lv2
```

如果 macOS 阻止打开下载的插件，可以对插件 bundle 执行上面的命令来移除 quarantine 属性。  
If macOS blocks a downloaded plugin, you can run the commands above on the plugin bundle to remove the quarantine attribute.

## Build / 构建

```bash
git clone --recurse https://github.com/ManasWolrd/VitalReverb.git

# Windows
cmake -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --config Release

# Linux
sudo apt update
sudo apt-get install libx11-dev libfreetype-dev libfontconfig1-dev libasound2-dev libxrandr-dev libxinerama-dev libxcursor-dev
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -S . -B .build
cmake --build build --config Release

# macOS
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -S . -B build
cmake --build build --config Release
```
