OPENGLCSFuzz - A Tool

---------------------

NOTE: This project uses Git submodules, be sure to use `git clone --recursive` when cloning or run `git submodule update --init --recursive` after you cloned it.

This is a tool designed to test OpenGL drivers, specifically the compute shader compiler + runtime. It generates a custom bytecode for a compute shader, and then converts that to a compute shader. It runs the compute shader on an input image, and collects the output. It then also simulates that bytecode on the CPU, and compares the two results.

It's uh...kinda finnicky right now. I've had to try to work around floating point imprecision and optimization, both of which make the exact tests done here very difficult. So I've made some sacrifices.

Still very WIP

This project uses:
 - [Dear ImGui, by Omar](https://github.com/ocornut/imgui)
 - [GLFW](https://github.com/glfw/glfw)
 - Some of [the STB libraries, by Sean Barret](https://github.com/nothings/stb)

Also, I've tried very hard (read: not very hard) to make it easy to build. In theory, you should be able to open the .sln file in Visual Studio 2019, and build it for Debug x64. If it doesn't work, I'm sorry. I very much don't like the OpenGL development ecosystem on Windows.
