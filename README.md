# PathtracerCUDA
Simple pathtracer with quadrics written in C++ and CUDA. It has the following features:
- Several shape types (sphere, cylinder, disk, cone, paraboloid, quad, cube)
- Different material types (lambertian diffuse, GGX specular, mix of both)
- Monte Carlo Importance Sampling
- Bounding Volume Hierarchy with Surface Area Heuristic
- Support for simple textured scenes
- Support for HDRI environments
- Windowed or headless rendering
- Interactable camera in windowed mode
- Images can be saved as PNG or HDR file
- Object transformations (translation, rotation, scale)
- Scene descriptions are read from a JSON file

# Usage

PathtracerCUDA.exe \[options\] \<input file\>

Option | Description
------------ | -------------
-help               | Display available options
-w                  | Set width of output image
-h                  | Set height of output image
-spp                | Set number of samples per pixel
-window             | Shows a window and displays progressive rendering results
-enable_controls    | Enables camera controls. If this option is enabled, the result image can only be saved manually by pressing the P key. The image is then saved to the filepath specified by -o. -window must be set for this option
-o                  | Set filepath of output image. Unless -enable_controls is enabled, the image will be saved to this location once rendering finishes
-ohdr               | Save image as HDR instead of PNG

The \<input file\> is expected to be a json file describing the scene to be rendered. The project includes two example scenes (see screenshots below) that show what such a file looks like.

# Controls
- Right click + mouse rotates the camera.
- WASD moves the camera
- P to save a screenshot to the filepath indicated by the command line argument "-o"

# Requirements
- GPU with CUDA Compute Capability 3.0 or better

# How to build
The project comes as a Visual Studio 2019 solution and already includes all dependencies except for the CUDA SDK (version 10.2). The Application should be build as x64.

# Screenshots

Cornell Box (4096spp):
![Cornell Box](PathtracerCUDA/cornell_box_4096spp.png?raw=true "Cornell Box")

Scene with several hundred random shapes with random materials and a HDRI environment (4096spp):
![Scene with several hundred random shapes with random materials and a HDRI environment](PathtracerCUDA/generated_scene_4096spp.png?raw=true "Scene with several hundred random shapes with random materials and a HDRI environment")


# Credits
- HDRI Environment https://hdrihaven.com/hdri/?h=roof_garden
- glad https://glad.dav1d.de/
- GLFW https://www.glfw.org/
- stb_image, stb_image_write https://github.com/nothings/stb
- nlohmann json https://github.com/nlohmann/json

