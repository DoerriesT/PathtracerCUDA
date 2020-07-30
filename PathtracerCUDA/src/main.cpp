#include "Pathtracer.h"
#include <glad/glad.h>
#include <cassert>
#include "Utility.h"
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <curand_kernel.h>
#include "kernels/initRandState.h"
#include "kernels/trace.h"
#include "stb_image.h"
#include <cstdlib>
#include "Window.h"
#include "UserInput.h"
#include "Pathtracer.h"
#include "Camera.h"
#include <random>
#include <GLFW/glfw3.h>
#include <cassert>
#include "Utility.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include "stb_image_write.h"

struct Params
{
	unsigned int m_width = 1024;
	unsigned int m_height = 1024;
	unsigned int m_spp = 1024;
	const char *m_inputFilepath = nullptr;
	const char *m_outputFilepath = nullptr;
	bool m_showWindow = false;
	bool m_enableControls = false;
	bool m_outputHdr = false;
};

bool processArgs(int argc, char *argv[], Params &params)
{
	constexpr const char *helpOption = "-help";
	constexpr const char *widthOption = "-w";
	constexpr const char *heightOption = "-h";
	constexpr const char *sppOption = "-spp";
	constexpr const char *windowOption = "-window";
	constexpr const char *controlsOption = "-enable_controls";
	constexpr const char *outputOption = "-o";
	constexpr const char *outputHdrOption = "-ohdr";

	params = {};

	bool displayHelp = false;

	int i = 1;
	for (; i < argc;)
	{
		if (strcmp(argv[i], helpOption) == 0)
		{
			displayHelp = true;
			++i;
			continue;
		}
		else if (strcmp(argv[i], widthOption) == 0)
		{
			if (i + 1 < argc)
			{
				params.m_width = atoi(argv[i + 1]);
				if (params.m_width == 0)
				{
					printf("Invalid input for %s!\n", widthOption);
					displayHelp = true;
				}
			}
			else
			{
				printf("Missing argument to %s!\n", widthOption);
				displayHelp = true;
			}
			i += 2;
			continue;
		}
		else if (strcmp(argv[i], heightOption) == 0)
		{
			if (i + 1 < argc)
			{
				params.m_height = atoi(argv[i + 1]);
				if (params.m_height == 0)
				{
					printf("Invalid input for %s!\n", heightOption);
					displayHelp = true;
				}
			}
			else
			{
				printf("Missing argument to %s!\n", heightOption);
				displayHelp = true;
			}
			i += 2;
			continue;
		}
		else if (strcmp(argv[i], sppOption) == 0)
		{
			if (i + 1 < argc)
			{
				params.m_spp = atoi(argv[i + 1]);
				if (params.m_spp == 0)
				{
					printf("Invalid input for %s!\n", sppOption);
					displayHelp = true;
				}
			}
			else
			{
				printf("Missing argument to %s!\n", sppOption);
				displayHelp = true;
			}
			i += 2;
			continue;
		}
		else if (strcmp(argv[i], windowOption) == 0)
		{
			params.m_showWindow = true;
			++i;
			continue;
		}
		else if (strcmp(argv[i], controlsOption) == 0)
		{
			params.m_enableControls = true;
			++i;
			continue;
		}
		else if (strcmp(argv[i], outputHdrOption) == 0)
		{
			params.m_outputHdr = true;
			++i;
			continue;
		}
		else if (strcmp(argv[i], outputOption) == 0)
		{
			if (i + 1 < argc)
			{
				params.m_outputFilepath = argv[i + 1];
			}
			else
			{
				printf("Missing argument to %s!\n", outputOption);
				displayHelp = true;
			}
			i += 2;
			continue;
		}
		else
		{
			break;
		}
	}

	if (i < argc && argc > 1)
	{
		params.m_inputFilepath = argv[argc - 1];
	}
	else if (!(argc == 2 && strcmp(argv[1], helpOption) == 0))
	{
		printf("Missing input file argument!\n");
		displayHelp = true;
	}

	if (displayHelp)
	{
		printf("USAGE: pathtracer.exe [options] <input file>\n\n");
		printf("Options:\n");
		printf("%-30s Display available options\n", helpOption);
		printf("%-30s Set width of output image\n", widthOption);
		printf("%-30s Set height of output image\n", heightOption);
		printf("%-30s Set number of samples per pixel\n", sppOption);
		printf("%-30s Shows a window and displays progressive rendering results\n", windowOption);
		printf("%-30s Enables camera controls (WASD to move, RMB+Mouse to rotate). "
			"If this option is enabled, the result image can only be saved manually by pressing the P key. "
			"The image is then saved to the filepath specified by %s. %s must be set for this option\n", controlsOption, outputOption, windowOption);
		printf("%-30s Set filepath of output image\n", outputOption);
		printf("%-30s Save image as HDR instead of PNG\n", outputHdrOption);

		return false;
	}

	// controls can only be enabled if there is a window
	params.m_enableControls = params.m_enableControls && params.m_showWindow;

	return true;
}

Camera loadScene(Pathtracer &pathtracer, const Params &params)
{
	std::map<std::string, uint32_t> texturePathToHandle;

	auto getTextureHandle = [&](const std::string &filepath) -> uint32_t
	{
		if (filepath.empty())
		{
			return 0;
		}

		auto it = texturePathToHandle.find(filepath);
		if (it != texturePathToHandle.end())
		{
			return it->second;
		}
		else
		{
			uint32_t handle = pathtracer.loadTexture(filepath.c_str());
			texturePathToHandle[filepath] = handle;
			return handle;
		}
	};

	auto getString = [](const nlohmann::basic_json<> &object, const char *key, std::string &result) -> bool
	{
		if (object.contains(key) && object[key].is_string())
		{
			result = object[key].get<std::string>();
			return true;
		}
		return false;
	};

	auto getFloat = [](const nlohmann::basic_json<> &object, const char *key, float &result) -> bool
	{
		if (object.contains(key) && object[key].is_number_float())
		{
			result = object[key].get<float>();
			return true;
		}
		return false;
	};

	auto getVec3 = [](const nlohmann::basic_json<> &object, const char *key, vec3 &result) -> bool
	{
		if (object.contains(key) && object[key].is_array() && object[key].size() == 3)
		{
			result = vec3(object[key][0].get<float>(), object[key][1].get<float>(), object[key][2].get<float>());
			return true;
		}
		return false;
	};

	auto getObject = [](const nlohmann::basic_json<> &object, const char *key, nlohmann::basic_json<> &result) -> bool
	{
		if (object.contains(key) && object[key].is_object())
		{
			result = object[key];
			return true;
		}
		return false;
	};

	auto radians = [](float degree)
	{
		return degree * (1.0f / 180.0f) * 3.14159265358979323846f;
	};

	nlohmann::json jscene;
	{
		std::ifstream file(params.m_inputFilepath);
		if (file.is_open())
		{
			try
			{
				file >> jscene;
			}
			catch (nlohmann::detail::parse_error ex)
			{
				printf(ex.what());
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			printf("Failed to open input file: %s\n", params.m_inputFilepath);
			exit(EXIT_FAILURE);
		}
	}

	std::vector<CpuHittable> objects;

	if (jscene.contains("objects") && jscene["objects"].is_array())
	{
		objects.reserve(jscene["objects"].size());

		for (const auto &o : jscene["objects"])
		{

			HittableType hittableType = HittableType::SPHERE;
			vec3 position = 0.0f;
			vec3 rotation = 0.0f;
			vec3 scale = 1.0f;

			MaterialType materialType = MaterialType::LAMBERT;
			vec3 baseColor = 1.0f;
			vec3 emissive = 0.0f;
			float roughness = 0.5f;
			float metalness = 0.0f;
			uint32_t textureHandle = 0;


			// parse hittable data
			std::string jhittableType;
			if (getString(o, "type", jhittableType))
			{
				if (jhittableType == "SPHERE")
				{
					hittableType = HittableType::SPHERE;
				}
				else if (jhittableType == "CYLINDER")
				{
					hittableType = HittableType::CYLINDER;
				}
				else if (jhittableType == "DISK")
				{
					hittableType = HittableType::DISK;
				}
				else if (jhittableType == "CONE")
				{
					hittableType = HittableType::CONE;
				}
				else if (jhittableType == "PARABOLOID")
				{
					hittableType = HittableType::PARABOLOID;
				}
				else if (jhittableType == "QUAD")
				{
					hittableType = HittableType::QUAD;
				}
				else if (jhittableType == "CUBE")
				{
					hittableType = HittableType::CUBE;
				}
				else
				{
					printf("Failed to parse object type: %s\n", jhittableType.c_str());
				}
			}
			getVec3(o, "position", position);
			getVec3(o, "rotation", rotation);
			getVec3(o, "scale", scale);


			// parse material data
			nlohmann::json m;
			if (getObject(o, "material", m))
			{
				std::string jmaterialType;
				if (getString(m, "type", jmaterialType))
				{
					if (jmaterialType == "LAMBERT")
					{
						materialType = MaterialType::LAMBERT;
					}
					else if (jmaterialType == "GGX")
					{
						materialType = MaterialType::GGX;
					}
					else if (jmaterialType == "LAMBERT_GGX")
					{
						materialType = MaterialType::LAMBERT_GGX;
					}
					else
					{
						printf("Failed to parse material type: %s\n", jmaterialType.c_str());
					}
				}

				getVec3(m, "baseColor", baseColor);
				getVec3(m, "emissive", emissive);
				getFloat(m, "roughness", roughness);
				getFloat(m, "metalness", metalness);
				std::string texturePath;
				if (getString(m, "texture", texturePath))
				{
					textureHandle = getTextureHandle(texturePath);
				}
			}

			objects.push_back(CpuHittable(hittableType, position, vec3(radians(rotation.x), radians(rotation.y), radians(rotation.z)), scale, Material2(materialType, baseColor, emissive, roughness, metalness, textureHandle)));
		}

		pathtracer.setScene(objects.size(), objects.data());
	}

	std::string skyboxTexturePath;
	if (getString(jscene, "skybox", skyboxTexturePath))
	{
		pathtracer.setSkyboxTextureHandle(getTextureHandle(skyboxTexturePath));
	}


	vec3 position = 0.0f;
	vec3 look_at = vec3(0.0f, 0.0f, -1.0f);
	float fovy = 60.0f;

	nlohmann::json c;
	if (getObject(jscene, "camera", c))
	{
		getVec3(c, "position", position);
		getVec3(c, "look_at", look_at);
		getFloat(c, "fovy", fovy);
	}

	return Camera(position, look_at, vec3(0.0f, 1.0f, 0.0f), radians(fovy), (float)params.m_width / params.m_height, 0.0f, 10.0f);
}

Camera setupScene(Pathtracer &pathtracer, const Params &params)
{
	uint32_t skyboxTextureHandle = pathtracer.loadTexture("skybox.hdr");
	uint32_t earthTextureHandle = pathtracer.loadTexture("earth.png");

	pathtracer.setSkyboxTextureHandle(skyboxTextureHandle);

	auto radians = [](float degree)
	{
		return degree * (1.0f / 180.0f) * 3.14159265358979323846f;
	};

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	float dist_to_focus = 10.0f;
	float aperture = 0.0f;
	float aspectRatio = (float)params.m_width / params.m_height;

	Camera camera(lookfrom, lookat, vup, radians(60.0f), aspectRatio, aperture, dist_to_focus);

	// create entities
	{
		std::default_random_engine e;
		std::uniform_real_distribution<float> d(0.0f, 1.0f);

		std::vector<CpuHittable> hittablesCpu;
		hittablesCpu.reserve(22 * 22 + 4);

		// cornell box
		//{
		//	// right side
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(1.0f, 1.0f, 0.0f), vec3(0.0f, radians(-90.0f), radians(-90.0f)), vec3(1.0f), Material2(vec3(0.0f, 1.0f, 0.0f))));
		//
		//	// left side
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(-1.0f, 1.0f, 0.0f), vec3(0.0f, radians(90.0f), radians(90.0f)), vec3(1.0f), Material2(vec3(1.0f, 0.0f, 0.0f))));
		//
		//	// back
		//	hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 1.0f, -1.0f), vec3(radians(-90.0f), 0.0f, 0.0f), vec3(1.0f), Material2(vec3(1.0f, 1.0f, 1.0f))));
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

		hittablesCpu.push_back(CpuHittable(HittableType::QUAD, vec3(0.0f, 0.0f, 0.0f), vec3(), vec3(20.0f), Material2(MaterialType::LAMBERT, vec3(1.0f), vec3(0.0f), 1.0f, 0.0f)));
		//hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, vec3(0.0f, -1000.0f, 0.0f), vec3(), vec3(1000.0f), Material2(vec3(0.5f))));// Material(Material::Type::LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f))));
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
					float metalness = d(e) > 0.5f ? 1.0f : 0.0f;
					hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, center, vec3(), vec3(0.2f), Material2(MaterialType::LAMBERT_GGX, albedo, 0.0f, 0.1f, metalness)));
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

		hittablesCpu.push_back(CpuHittable(HittableType::SPHERE, vec3(0.0f, 1.0f, 0.0f), vec3(), vec3(1.0f), Material2(MaterialType::LAMBERT_GGX, vec3(0.5f), 0.0f, 0.5f, 0.0f, earthTextureHandle)));

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

		pathtracer.setScene(hittablesCpu.size(), hittablesCpu.data());
	}

	return camera;
}

void saveImage(const Params &params, Pathtracer &pathtracer)
{
	printf("Writing result to %s\n", params.m_outputFilepath);

	stbi_flip_vertically_on_write(true);
	if (params.m_outputHdr)
	{
		if (!stbi_write_hdr(params.m_outputFilepath, params.m_width, params.m_height, 4, pathtracer.getHDRImageData()))
		{
			printf("Failed to write file!\n");
		}
	}
	else
	{
		if (!stbi_write_png(params.m_outputFilepath, params.m_width, params.m_height, 4, pathtracer.getImageData(), params.m_width * 4))
		{
			printf("Failed to write file!\n");
		}
	}
}

int main(int argc, char *argv[])
{
	// fill parameter struct with command line arguments.
	// exit if help text was displayed.
	Params params;
	if (!processArgs(argc, argv, params))
	{
		return EXIT_SUCCESS;
	}

	printf("Beginning rendering in configuration:\n");
	printf("Width: %d\n", (int)params.m_width);
	printf("Height: %d\n", (int)params.m_height);
	printf("Samples per Pixel: %d\n", (int)params.m_spp);
	printf("Window: %d\n", (int)params.m_showWindow);
	printf("Controls: %d\n", (int)params.m_enableControls);
	printf("Output HDR: %d\n", (int)params.m_outputHdr);
	printf("Output Filepath: %s\n", params.m_outputFilepath ? params.m_outputFilepath : "");
	printf("Input Filepath: %s\n", params.m_inputFilepath);

	Window *window = nullptr;
	UserInput *userInput = nullptr;
	GLuint pixelBufferGL = 0;

	// optionally create a window
	if (params.m_showWindow)
	{
		window = new Window(params.m_width, params.m_height, "CUDA Pathtracer");

		// init opengl
		{
			if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
			{
				Utility::fatalExit("Failed to initialize GLAD!", EXIT_FAILURE);
			}

			glViewport(0, 0, params.m_width, params.m_height);
			assert(glGetError() == GL_NO_ERROR);
			glGenBuffers(1, &pixelBufferGL);
			assert(glGetError() == GL_NO_ERROR);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferGL);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, params.m_width * params.m_height * 4, nullptr, GL_STREAM_DRAW);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			assert(glGetError() == GL_NO_ERROR);
		}
	}

	// optionally create a UserInput object
	if (params.m_showWindow && params.m_enableControls)
	{
		userInput = new UserInput();
		window->addInputListener(userInput);
	}

	// create Pathtracer instance
	Pathtracer pathtracer(params.m_width, params.m_height, pixelBufferGL);

	Camera camera = loadScene(pathtracer, params);

	if (!params.m_showWindow)
	{
		constexpr uint32_t samplesPerIteration = 8;
		float totalGpuTime = 0.0f;

		for (uint32_t i = 0; i < params.m_spp; i += samplesPerIteration)
		{
			const uint32_t spp = std::min(i + samplesPerIteration, params.m_spp) - i;
			pathtracer.render(camera, spp, i == 0);

			totalGpuTime += pathtracer.getTiming();

			if (i % (samplesPerIteration * 4) == 0)
			{
				printf("Accumulated %d samples\n", (int)i);
			}
		}

		printf("Finished accumulating %d samples in %f ms GPU time\n", (int)params.m_spp, totalGpuTime);
		
		// writing to file is optional
		if (params.m_outputFilepath)
		{
			saveImage(params, pathtracer);
		}
	}
	else
	{
		double lastTime = glfwGetTime();
		double timeDelta = 0.0;
		uint32_t accumulatedSamples = 0;
		bool grabbedMouse = false;
		bool savedFileAutomatically = false;
		float totalGpuTime = 0.0f;

		while (!window->shouldClose())
		{
			double time = glfwGetTime();
			timeDelta = time - lastTime;
			lastTime = time;
			window->pollEvents();

			bool resetAccumulation = false;
			bool saveToFile = false;

			// handle input
			if (userInput)
			{
				userInput->input();

				bool pressed = false;
				float mod = 5.0f;
				float cameraTranslation[3] = {};

				float mouseDelta[2] = {};

				if (userInput->isMouseButtonPressed(InputMouse::BUTTON_RIGHT))
				{
					if (!grabbedMouse)
					{
						grabbedMouse = true;
						window->grabMouse(grabbedMouse);
					}
					userInput->getMousePosDelta(mouseDelta[0], mouseDelta[1]);
				}
				else
				{
					if (grabbedMouse)
					{
						grabbedMouse = false;
						window->grabMouse(grabbedMouse);
					}
				}

				if (mouseDelta[0] * mouseDelta[0] + mouseDelta[1] * mouseDelta[1] > 0.0f)
				{
					camera.rotate(mouseDelta[1] * 0.005f, mouseDelta[0] * 0.005f, 0.0f);
					resetAccumulation = true;
				}


				if (userInput->isKeyPressed(InputKey::LEFT_SHIFT))
				{
					mod = 25.0f;
				}
				if (userInput->isKeyPressed(InputKey::W))
				{
					cameraTranslation[2] = -mod * (float)timeDelta;
					pressed = true;
				}
				if (userInput->isKeyPressed(InputKey::S))
				{
					cameraTranslation[2] = mod * (float)timeDelta;
					pressed = true;
				}
				if (userInput->isKeyPressed(InputKey::A))
				{
					cameraTranslation[0] = -mod * (float)timeDelta;
					pressed = true;
				}
				if (userInput->isKeyPressed(InputKey::D))
				{
					cameraTranslation[0] = mod * (float)timeDelta;
					pressed = true;
				}
				if (pressed)
				{
					camera.translate(cameraTranslation[0], cameraTranslation[1], cameraTranslation[2]);
					resetAccumulation = true;
				}

				if (userInput->isKeyPressed(InputKey::P, true))
				{
					saveToFile = true;
				}
			}


			// render frame
			if (accumulatedSamples < params.m_spp)
			{
				pathtracer.render(camera, 1, resetAccumulation);
				++accumulatedSamples;

				totalGpuTime += pathtracer.getTiming();
				if (accumulatedSamples == params.m_spp)
				{
					printf("Finished accumulating %d samples in %f ms GPU time\n", (int)params.m_spp, totalGpuTime);
				}
			}

			// copy to backbuffer
			glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glRasterPos2i(-1, -1);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferGL);
			glDrawPixels(params.m_width, params.m_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			assert(glGetError() == GL_NO_ERROR);

			accumulatedSamples = resetAccumulation ? 0 : accumulatedSamples;
			resetAccumulation = false;

			window->present();
			window->setTitle("Accumulated Samples: " + std::to_string(accumulatedSamples) + " Frametime: " + std::to_string(pathtracer.getTiming()) + " ms");


			// writing to file is optional
			if ((saveToFile || (accumulatedSamples == params.m_spp && !params.m_enableControls && !savedFileAutomatically)) && params.m_outputFilepath)
			{
				saveToFile = false;
				savedFileAutomatically = true;
				saveImage(params, pathtracer);
			}
		}

		// delete pixel buffer object
		glDeleteBuffers(1, &pixelBufferGL);
		assert(glGetError() == GL_NO_ERROR);
	}

	return EXIT_SUCCESS;
}