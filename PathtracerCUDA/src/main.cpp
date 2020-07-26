#include <cstdlib>
#include "Window.h"
#include "UserInput.h"
#include "Pathtracer.h"
#include "Camera.h"
#include <random>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>

int main()
{
	Window window(1600, 900, "Pathtracer CUDA");
	UserInput input;
	bool grabbedMouse = false;
	window.addInputListener(&input);

	uint32_t width = window.getWidth();
	uint32_t height = window.getHeight();

	Pathtracer pathtracer(width, height);

	auto radians = [](float degree)
	{
		return degree * (1.0f / 180.0f) * 3.14159265358979323846f;
	};

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	float dist_to_focus = 10.0f;
	float aperture = 0.0f;
	float aspectRatio = (float)width / height;

	Camera camera(lookfrom, lookat, vup, radians(60.0f), aspectRatio, aperture, dist_to_focus);

	BVH bvh;
	// create entities
	{
		std::default_random_engine e;
		std::uniform_real_distribution<float> d(0.0f, 1.0f);

		std::vector<CpuHittable> hittablesCpu;
		hittablesCpu.reserve(22 * 22 + 4);

		//// cornell box
		//{
		//	// right side
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(1.0f, 1.0f, 0.0f), vec3(0.0f, radians(-90.0f), radians(-90.0f)), vec3(1.0f), Material2(vec3(0.0f, 1.0f, 0.0f))));
		//
		//	// left side
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(-1.0f, 1.0f, 0.0f), vec3(0.0f, radians(90.0f), radians(90.0f)), vec3(1.0f), Material2(vec3(1.0f, 0.0f, 0.0f))));
		//
		//	// back
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 1.0f, -1.0f),vec3(radians(-90.0f), 0.0f, 0.0f), vec3(1.0f), Material2(vec3(1.0f, 1.0f, 1.0f))));
		//
		//	// top
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 2.0f, 0.0f), vec3(), vec3(1.0f), Material2(vec3(1.0f, 1.0f, 1.0f))));
		//
		//	// bottom
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 0.0f, 0.0f), vec3(), vec3(1.0f), Material2(vec3(1.0f, 1.0f, 1.0f))));
		//
		//	// light
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 1.99f, 0.0f), vec3(), vec3(0.3f), Material2(vec3(1.0f, 1.0f, 1.0f), vec3(10.0f))));
		//
		//	// big box
		//	hittablesCpu.push_back(CpuHittable(HittableType::CUBE, vec3(-0.5f, 0.75f, -0.5f), vec3(0.0f, radians(40.0f), 0.0f), vec3(0.25f, 0.75f, 0.25f), Material2(vec3(1.0f, 1.0f, 1.0f))));
		//	
		//	// small box
		//	hittablesCpu.push_back(CpuHittable(HittableType::CUBE, vec3(0.5f, 0.25f, 0.5f), vec3(0.0f, radians(-30.0f), 0.0f), vec3(0.25f), Material2(vec3(1.0f, 1.0f, 1.0f))));
		//}

		hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, vec3(0.0f, -1000.0f, 0.0f), vec3(), vec3(1000.0f), Material2(vec3(0.5f))));// Material(Material::Type::LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f))));
		//hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 0.0f, 0.0f), vec3(), vec3(100.0f, 1.0f, 100.0f), Material2(vec3(0.5f))));// Material(Material::Type::LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f))));

		
		for (int a = -11; a < 11; ++a)
		{
			for (int b = -11; b < 11; ++b)
			{
				auto chooseMat = d(e);
				vec3 center(a + 0.9f * d(e), 0.2f, b + 0.9f * d(e));
				if (length(center - vec3(4.0f, 0.2f, 0.0f)) > 0.9f)
				{
					auto albedo = vec3(d(e), d(e), d(e)) * vec3(d(e), d(e), d(e));
					hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, center, vec3(), vec3(0.2f), Material2(albedo)));
					//if (chooseMat < 0.8f)
					//{
					//	// diffuse
					//	auto albedo = vec3(d(e), d(e), d(e)) * vec3(d(e), d(e), d(e));
					//	hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, CpuHittable::Payload(Sphere{ center, 0.2f }), Material(Material::Type::LAMBERTIAN, albedo)));
					//}
					//else if (chooseMat < 0.95f)
					//{
					//	// metal
					//	auto albedo = vec3(d(e), d(e), d(e)) * 0.5f + 0.5f;
					//	auto fuzz = d(e) * 0.5f;
					//	hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, CpuHittable::Payload(Sphere{ center, 0.2f }), Material(Material::Type::METAL, albedo, fuzz)));
					//}
					//else
					//{
					//	// glass
					//	hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, CpuHittable::Payload(Sphere{ center, 0.2f }), Material(Material::Type::DIELECTRIC, vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
					//}
				}
			}
		}

		//hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, CpuHittable::Payload(Sphere{ vec3(0.0f, 1.0f, 0.0f), 1.0f }), Material(Material::Type::DIELECTRIC, vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
		//
		//hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, CpuHittable::Payload(Sphere{ vec3(-4.0f, 1.0f, 0.0f), 1.0f }), Material(Material::Type::LAMBERTIAN, vec3(0.4f, 0.2f, 0.1f))));
		//
		//hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, CpuHittable::Payload(Sphere{ vec3(4.0f, 1.0f, 0.0f), 1.0f }), Material(Material::Type::METAL, vec3(0.7f, 0.6f, 0.5f), 0.0f)));
		//
		//hittablesCpu.push_back(CpuHittable(HittableType::CYLINDER, CpuHittable::Payload(Cylinder{ vec3(10.0f, 1.0f, 10.0f), 1.0f, 1.0f }), Material(Material::Type::DIELECTRIC, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f)));
		//
		//hittablesCpu.push_back(CpuHittable(HittableType::DISK, CpuHittable::Payload(Disk{ vec3(10.0f, 2.0f, 10.0f), 1.0f }), Material(Material::Type::LAMBERTIAN, vec3(1.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
		//
		//hittablesCpu.push_back(CpuHittable(HittableType::CONE, CpuHittable::Payload(Cone{ vec3(0.0f, 3.0f, 0.0f), 1.0f, 1.0f }), Material(Material::Type::LAMBERTIAN, vec3(1.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
		//
		//hittablesCpu.push_back(CpuHittable(HittableType::PARABOLOID, CpuHittable::Payload(Paraboloid{ vec3(3.0f, 1.0f, 3.0f), 1.0f, 3.0f }), Material(Material::Type::METAL, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f)));


		bvh.build(hittablesCpu.size(), hittablesCpu.data(), 4);
		assert(bvh.validate());
	}

	// translate hittables to their cpu version
	auto &bvhElements = bvh.getElements();
	std::vector<Hittable> gpuHittables;
	gpuHittables.reserve(bvh.getElements().size());
	for (const auto &e : bvhElements)
	{
		gpuHittables.push_back(e.getGpuHittable());
	}

	pathtracer.setBVH((uint32_t)bvh.getNodes().size(), bvh.getNodes().data(), (uint32_t)gpuHittables.size(), gpuHittables.data());

	double lastTime = glfwGetTime();
	double timeDelta = 0.0;
	uint32_t frame = 0;
	while (!window.shouldClose())
	{
		double time = glfwGetTime();
		timeDelta = time - lastTime;
		lastTime = time;
		window.pollEvents();
		input.input();

		bool resetAccumulation = false;

		// handle input
		{
			bool pressed = false;
			float mod = 5.0f;
			glm::vec3 cameraTranslation;

			glm::vec2 mouseDelta = {};

			if (input.isMouseButtonPressed(InputMouse::BUTTON_RIGHT))
			{
				if (!grabbedMouse)
				{
					grabbedMouse = true;
					window.grabMouse(grabbedMouse);
				}
				mouseDelta = input.getMousePosDelta();
			}
			else
			{
				if (grabbedMouse)
				{
					grabbedMouse = false;
					window.grabMouse(grabbedMouse);
				}
			}

			if (mouseDelta.x * mouseDelta.x + mouseDelta.y * mouseDelta.y > 0.0f)
			{
				camera.rotate(mouseDelta.y * 0.005f, mouseDelta.x * 0.005f, 0.0f);
				resetAccumulation = true;
			}


			if (input.isKeyPressed(InputKey::LEFT_SHIFT))
			{
				mod = 25.0f;
			}
			if (input.isKeyPressed(InputKey::W))
			{
				cameraTranslation.z = -mod * (float)timeDelta;
				pressed = true;
			}
			if (input.isKeyPressed(InputKey::S))
			{
				cameraTranslation.z = mod * (float)timeDelta;
				pressed = true;
			}
			if (input.isKeyPressed(InputKey::A))
			{
				cameraTranslation.x = -mod * (float)timeDelta;
				pressed = true;
			}
			if (input.isKeyPressed(InputKey::D))
			{
				cameraTranslation.x = mod * (float)timeDelta;
				pressed = true;
			}
			if (pressed)
			{
				camera.translate(cameraTranslation.x, cameraTranslation.y, cameraTranslation.z);
				resetAccumulation = true;
			}
		}


		// render frame
		pathtracer.render(camera, resetAccumulation);

		frame = resetAccumulation ? 0 : frame;
		resetAccumulation = false;

		window.present();
		window.setTitle("Accumulated Samples: " + std::to_string(frame) + " Frametime: " + std::to_string(pathtracer.getTiming()) + " ms");
		++frame;
	}

	return EXIT_SUCCESS;
}