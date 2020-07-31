#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "util/Window.h"
#include "util/UserInput.h"
#include "pathtracer/Camera.h"
#include "pathtracer/Pathtracer.h"
#include "Params.h"
#include "SceneLoader.h"

#define GENERATE_SCENE_FILE 0

#ifndef GENERATE_SCENE_FILE
#define GENERATE_SCENE_FILE 0
#endif // GENERATE_SCENE_FILE


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

	// initialize params with default values
	params = {};

	bool displayHelp = false;

	// walk through command line arguments and evaluate them
	int i = 1;
	for (; i < (argc - 1);)
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
			printf("Can't parse argument: %s\n", argv[i]);
			displayHelp = true;
			break;
		}
	}

	// last argument should be filepath to scene file
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
		printf("USAGE: PathtracerCUDA.exe [options] <input file>\n\n");
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
#if GENERATE_SCENE_FILE
	generateSceneFile();
	return EXIT_SUCCESS;
#endif // GENERATE_SCENE_FILE

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
				printf("Failed to initialize GLAD!\n");
				return EXIT_FAILURE;
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

	// headless rendering: do not open a window and simply accumulate params.m_spp samples and optionally
	// save the resulting image
	if (!params.m_showWindow)
	{
		// we cant compute all samples in one go or we risk getting a TDR on slower machines
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
	// rendering with window: present each iteration to the user and optionally allow manipulating the camera and saving screenshots.
	// of controls are disabled, we optionally save the resulting image once all samples are accumulated
	else
	{
		double lastTime = glfwGetTime();
		double timeDelta = 0.0;
		uint32_t accumulatedSamples = 0;
		bool grabbedMouse = false;
		bool savedFileAutomatically = false;
		float totalGpuTime = 0.0f;

		// we keep the window open even after having accumulated all samples
		while (!window->shouldClose())
		{
			double time = glfwGetTime();
			timeDelta = time - lastTime;
			lastTime = time;
			window->pollEvents();

			bool resetAccumulation = false;
			bool saveToFile = false;

			// handle input (optional)
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

				// reset accumulation if camera was rotatd
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

				// reset accumulation if camera was moved
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