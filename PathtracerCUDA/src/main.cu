#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cassert>
#include "Utility.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_gl_interop.h"
#include <curand_kernel.h>

#include <stdio.h>

#include "Window.h"
#include "UserInput.h"
#include <iostream>

#include "Ray.h"
#include "vec3.h"
#include "Hittable.h"
#include "Camera.h"
#include <random>
#include "BVH.h"
#include "HitRecord.h"
#include "Material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ bool hitList(uint32_t hittableCount, Hittable *world, const Ray &r, float t_min, float t_max, HitRecord &rec)
{
	HitRecord tempRec;
	bool hitAnything = false;
	auto closest = t_max;

	for (uint32_t i = 0; i < hittableCount; ++i)
	{
		if (world[i].hit(r, t_min, closest, tempRec))
		{
			hitAnything = true;
			closest = tempRec.m_t;
			rec = tempRec;
		}
	}

	return hitAnything;
}

__device__ bool hitBVH(uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, const Ray &r, float t_min, float t_max, HitRecord &rec)
{
	vec3 invRayDir;
	invRayDir[0] = 1.0f / (r.m_dir[0] != 0.0f ? r.m_dir[0] : pow(2.0f, -80.0f));
	invRayDir[1] = 1.0f / (r.m_dir[1] != 0.0f ? r.m_dir[1] : pow(2.0f, -80.0f));
	invRayDir[2] = 1.0f / (r.m_dir[2] != 0.0f ? r.m_dir[2] : pow(2.0f, -80.0f));
	vec3 originDivDir = r.m_origin * invRayDir;
	bool dirIsNeg[3] = { invRayDir.x < 0.0f, invRayDir.y < 0.0f, invRayDir.z < 0.0f };

	uint32_t nodesToVisit[32];
	uint32_t toVisitOffset = 0;
	uint32_t currentNodeIndex = 0;

	uint32_t elemIdx = UINT32_MAX;
	uint32_t iterations = 0;

	while (true)
	{
		++iterations;
		const BVHNode &node = bvhNodes[currentNodeIndex];
		if (node.m_aabb.hit(r, t_min, t_max))
		{
			const uint32_t primitiveCount = (node.m_primitiveCountAxis >> 16);
			if (primitiveCount > 0)
			{
				for (uint32_t i = 0; i < primitiveCount; ++i)
				{
					Hittable &elem = world[node.m_offset + i];
					if (elem.hit(r, t_min, t_max, rec))
					{
						t_max = rec.m_t;
						elemIdx = node.m_offset + i;
					}
				}
				if (toVisitOffset == 0)
				{
					break;
				}
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else
			{
				if (dirIsNeg[(node.m_primitiveCountAxis >> 8) & 0xFF])
				{
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node.m_offset;
				}
				else
				{
					nodesToVisit[toVisitOffset++] = node.m_offset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else
		{
			if (toVisitOffset == 0)
			{
				break;
			}
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	return elemIdx != UINT32_MAX;
}

__device__ vec3 getColor(const Ray &r, uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, curandState &randState)
{
	vec3 beta = vec3(1.0f);
	vec3 L = vec3(0.0f);
	Ray ray = r;
	for (int iteration = 0; iteration < 5; ++iteration)
	{
		HitRecord rec;
		bool foundIntersection = hitBVH(hittableCount, world, bvhNodesCount, bvhNodes, ray, 0.001f, FLT_MAX, rec);

		// add emitted light
		if (foundIntersection)
		{
			//L += beta * rec.m_emitted;
		}

		// add sky light and exit loop
		if (!foundIntersection)
		{
			vec3 unitDir = normalize(ray.m_dir);
			float t = unitDir.y * 0.5f + 0.5f;
			vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
			L += beta * c;
			break;
		}

		// scatter
		{
			Ray scattered;
			float pdf = 0.0f;
			vec3 attenuation = rec.m_material->sample(ray, rec, randState, scattered, pdf);
			if (attenuation == vec3(0.0f) || pdf == 0.0f)
			{
				break;
			}
			beta *= attenuation * abs(dot(normalize(scattered.m_dir), normalize(rec.m_normal))) / pdf;
			ray = scattered;
		}

		//if (hitBVH(hittableCount, world, bvhNodesCount, bvhNodes, ray, 0.001f, FLT_MAX, rec))
		////if (hitList(hittableCount, world, ray, 0.001f, FLT_MAX, rec))
		//{
		//	Ray scattered;
		//	vec3 attenuation;
		//	if (rec.m_material->scatter(ray, rec, randState, attenuation, scattered))
		//	{
		//		beta *= attenuation;
		//		ray = scattered;
		//	}
		//	else
		//	{
		//		return vec3(0.0f, 0.0f, 0.0f);
		//	}
		//}
		//else
		//{
		//	vec3 unitDir = normalize(ray.m_dir);
		//	float t = unitDir.y * 0.5f + 0.5f;
		//	vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
		//	return c * beta;
		//}
	}

	return L;
}

__global__ void traceKernel(uchar4 *resultBuffer, float4 *accumBuffer, bool ignoreHistory, uint32_t frame, uint32_t width, uint32_t height, uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, curandState *randState, Camera camera)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	const uint32_t dstIdx = threadIDx + threadIDy * width;

	float4 inputColor4 = accumBuffer[dstIdx];
	vec3 inputColor(inputColor4.x, inputColor4.y, inputColor4.z);

	curandState &localRandState = randState[dstIdx];

	float u = (threadIDx + curand_uniform(&localRandState)) / float(width);
	float v = (threadIDy + curand_uniform(&localRandState)) / float(height);
	Ray r = camera.getRay(u, v, localRandState);
	vec3 color = getColor(r, hittableCount, world, bvhNodesCount, bvhNodes, localRandState);

	color = ignoreHistory ? color : color + inputColor;

	vec3 resultColor = color / float(frame + 1.0f);
	resultColor = saturate(resultColor);
	resultColor.r = sqrt(resultColor.r);
	resultColor.g = sqrt(resultColor.g);
	resultColor.b = sqrt(resultColor.b);

	accumBuffer[dstIdx] = { color.r, color.g, color.b, 1.0f };
	resultBuffer[dstIdx] = { (unsigned char)(resultColor.x * 255.0f), (unsigned char)(resultColor.y * 255.0f) , (unsigned char)(resultColor.z * 255.0f), 255 };
}

__global__ void initRandState(int width, int height, curandState *randState)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	uint32_t dstIdx = threadIDx + threadIDy * width;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984 + dstIdx, 0, 0, &randState[dstIdx]);
}

int main()
{
	Window window(1600, 900, "Pathtracer CUDA");
	UserInput input;
	bool grabbedMouse = false;
	window.addInputListener(&input);

	uint32_t width = window.getWidth();
	uint32_t height = window.getHeight();

	GLuint pixelBufferGL = 0;
	cudaGraphicsResource *pixelBufferCuda = nullptr;
	float4 *accumBuffer = nullptr;
	Hittable *hittables;
	BVHNode *bvhNodes;
	curandState *d_randState;
	uint32_t entityListSize = 22 * 22 + 4;
	uint32_t hittablesCount = 0;
	uint32_t bvhNodesCount = 0;

	cudaEvent_t startEvent, stopEvent;

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

	// init opengl
	{
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			Utility::fatalExit("Failed to initialize GLAD!", EXIT_FAILURE);
		}

		glViewport(0, 0, width, height);
		assert(glGetError() == GL_NO_ERROR);
		glGenBuffers(1, &pixelBufferGL);
		assert(glGetError() == GL_NO_ERROR);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferGL);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		assert(glGetError() == GL_NO_ERROR);
	}

	// init cuda
	{
		checkCudaErrors(cudaSetDevice(0));
		// register with cuda
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&pixelBufferCuda, pixelBufferGL, cudaGraphicsMapFlagsWriteDiscard));

		// alloc memory for prng state
		checkCudaErrors(cudaMalloc((void **)&d_randState, width * height * sizeof(curandState)));

		// init prng state
		dim3 threads(8, 8, 1);
		dim3 blocks((width + 7) / 8, (height + 7) / 8, 1);
		initRandState << <blocks, threads >> > (width, height, d_randState);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// alloc memory for accum buffer
		checkCudaErrors(cudaMalloc((void **)&accumBuffer, width * height * sizeof(float4)));

		checkCudaErrors(cudaEventCreate(&startEvent));
		checkCudaErrors(cudaEventCreate(&stopEvent));

		// create entities
		{
			std::default_random_engine e;
			std::uniform_real_distribution<float> d(0.0f, 1.0f);

			std::vector<Hittable> hittablesCpu;
			hittablesCpu.reserve(entityListSize);

			hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ vec3(0.0f, -1000.0f, 0.0f), 1000.0f }), Material2(vec3(0.5f))));// Material(Material::Type::LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f))));

			for (int a = -11; a < 11; ++a)
			{
				for (int b = -11; b < 11; ++b)
				{
					auto chooseMat = d(e);
					vec3 center(a + 0.9f * d(e), 0.2f, b + 0.9f * d(e));
					if (length(center - vec3(4.0f, 0.2f, 0.0f)) > 0.9f)
					{
						auto albedo = vec3(d(e), d(e), d(e)) * vec3(d(e), d(e), d(e));
						hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ center, 0.2f }), Material2(albedo)));
						//if (chooseMat < 0.8f)
						//{
						//	// diffuse
						//	auto albedo = vec3(d(e), d(e), d(e)) * vec3(d(e), d(e), d(e));
						//	hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ center, 0.2f }), Material(Material::Type::LAMBERTIAN, albedo)));
						//}
						//else if (chooseMat < 0.95f)
						//{
						//	// metal
						//	auto albedo = vec3(d(e), d(e), d(e)) * 0.5f + 0.5f;
						//	auto fuzz = d(e) * 0.5f;
						//	hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ center, 0.2f }), Material(Material::Type::METAL, albedo, fuzz)));
						//}
						//else
						//{
						//	// glass
						//	hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ center, 0.2f }), Material(Material::Type::DIELECTRIC, vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
						//}
					}
				}
			}

			//hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ vec3(0.0f, 1.0f, 0.0f), 1.0f }), Material(Material::Type::DIELECTRIC, vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
			//
			//hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ vec3(-4.0f, 1.0f, 0.0f), 1.0f }), Material(Material::Type::LAMBERTIAN, vec3(0.4f, 0.2f, 0.1f))));
			//
			//hittablesCpu.push_back(Hittable(Hittable::Type::SPHERE, Hittable::Payload(Sphere{ vec3(4.0f, 1.0f, 0.0f), 1.0f }), Material(Material::Type::METAL, vec3(0.7f, 0.6f, 0.5f), 0.0f)));
			//
			//hittablesCpu.push_back(Hittable(Hittable::Type::CYLINDER, Hittable::Payload(Cylinder{ vec3(10.0f, 1.0f, 10.0f), 1.0f, 1.0f }), Material(Material::Type::DIELECTRIC, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f)));
			//
			//hittablesCpu.push_back(Hittable(Hittable::Type::DISK, Hittable::Payload(Disk{ vec3(10.0f, 2.0f, 10.0f), 1.0f }), Material(Material::Type::LAMBERTIAN, vec3(1.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
			//
			//hittablesCpu.push_back(Hittable(Hittable::Type::CONE, Hittable::Payload(Cone{ vec3(0.0f, 3.0f, 0.0f), 1.0f, 1.0f }), Material(Material::Type::LAMBERTIAN, vec3(1.0f, 0.0f, 0.0f), 0.0f, 1.5f)));
			//
			//hittablesCpu.push_back(Hittable(Hittable::Type::PARABOLOID, Hittable::Payload(Paraboloid{ vec3(3.0f, 1.0f, 3.0f), 1.0f, 3.0f }), Material(Material::Type::METAL, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f)));


			bvh.build(hittablesCpu.size(), hittablesCpu.data(), 4);
			assert(bvh.validate());
			hittablesCpu = bvh.getElements();
			const auto &bvhNodesCpu = bvh.getNodes();

			checkCudaErrors(cudaMalloc((void **)&hittables, hittablesCpu.size() * sizeof(Hittable)));
			checkCudaErrors(cudaMalloc((void **)&bvhNodes, bvhNodesCpu.size() * sizeof(BVHNode)));

			checkCudaErrors(cudaMemcpy(hittables, hittablesCpu.data(), hittablesCpu.size() * sizeof(Hittable), cudaMemcpyKind::cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(bvhNodes, bvhNodesCpu.data(), bvhNodesCpu.size() * sizeof(BVHNode), cudaMemcpyKind::cudaMemcpyHostToDevice));

			hittablesCount = (uint32_t)hittablesCpu.size();
			bvhNodesCount = (uint32_t)bvhNodesCpu.size();
		}
	}

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

		frame = resetAccumulation ? 0 : frame;
		resetAccumulation = false;

		uchar4 *deviceMem;
		size_t numBytes;
		checkCudaErrors(cudaGraphicsMapResources(1, &pixelBufferCuda));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&deviceMem, &numBytes, pixelBufferCuda));

		// do something with cuda
		checkCudaErrors(cudaEventRecord(startEvent));
		dim3 threads(8, 8, 1);
		dim3 blocks((width + 7) / 8, (height + 7) / 8, 1);
		traceKernel << <blocks, threads >> > (deviceMem, accumBuffer, frame == 0, frame, width, height, hittablesCount, hittables, bvhNodesCount, bvhNodes, d_randState, camera);
		checkCudaErrors(cudaEventRecord(stopEvent));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaEventSynchronize(stopEvent));
		float milliseconds = 0;
		checkCudaErrors(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

		checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBufferCuda));

		// render
		glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glRasterPos2i(-1, -1);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferGL);
		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		assert(glGetError() == GL_NO_ERROR);

		window.present();
		window.setTitle(std::to_string(frame) + " Frametime: " + std::to_string(milliseconds));
		++frame;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	// unregister pixel buffer
	checkCudaErrors(cudaGraphicsUnregisterResource(pixelBufferCuda));

	// free cuda memory
	checkCudaErrors(cudaFree(hittables));
	checkCudaErrors(cudaFree(bvhNodes));
	checkCudaErrors(cudaFree(d_randState));

	// delete pixel buffer object
	glDeleteBuffers(1, &pixelBufferGL);
	assert(glGetError() == GL_NO_ERROR);

	return EXIT_SUCCESS;
}


