#pragma once
#include "pathtracer/Camera.h"

class Pathtracer;
struct Params;

// generates a scene file (generated_scene.json) with a few hundred random objects and a skybox
void generateSceneFile();

Camera loadScene(Pathtracer &pathtracer, const Params &params);