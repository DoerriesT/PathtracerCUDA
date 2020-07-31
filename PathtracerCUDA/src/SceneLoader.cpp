#include "SceneLoader.h"
#include "Params.h"
#include "pathtracer/Pathtracer.h"
#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include <iomanip>

// generates a scene file (generated_scene.json) with a few hundred random objects and a skybox
void generateSceneFile()
{
	nlohmann::json jscene;

	jscene["camera"] =
	{
		{"position", {13.0f, 2.0f, 3.0f}},
		{"look_at", {0.0f, 0.0f, 0.0f}},
		{"fovy", 60.0f},
	};

	jscene["skybox"] = "skybox.hdr";

	jscene["objects"] = nlohmann::json::array();
	{
		auto addObject = [](nlohmann::json &j,
			HittableType hittableType, const vec3 &position, const vec3 &rotation, const vec3 &scale,
			MaterialType materialType, const vec3 &baseColor, const vec3 &emissive, float roughness, float metalness, const char *texture)
		{
			auto hitToString = [](HittableType hittableType) -> const char *
			{
				switch (hittableType)
				{
				case HittableType::SPHERE:
					return "SPHERE";
				case HittableType::CYLINDER:
					return "CYLINDER";
				case HittableType::DISK:
					return "DISK";
				case HittableType::CONE:
					return "CONE";
				case HittableType::PARABOLOID:
					return "PARABOLOID";
				case HittableType::QUAD:
					return "QUAD";
				case HittableType::CUBE:
					return "CUBE";
				default:
					assert(false);
					break;
				}
				return "SPHERE";
			};

			auto matToString = [](MaterialType materialType) -> const char *
			{
				switch (materialType)
				{
				case MaterialType::LAMBERT:
					return "LAMBERT";
				case MaterialType::GGX:
					return "GGX";
				case MaterialType::LAMBERT_GGX:
					return "LAMBERT_GGX";
				default:
					assert(false);
					break;
				}
				return "LAMBERT";
			};

			j.push_back(
				{
					{"name", ""},
					{"type", hitToString(hittableType) },
					{"position", {position.x, position.y, position.z} },
					{"rotation", {rotation.x, rotation.y, rotation.z} },
					{"scale", {scale.x, scale.y, scale.z} },
					{"material",
						{
							{"type", matToString(materialType)},
							{"baseColor", {baseColor.x, baseColor.y, baseColor.z}},
							{"emissive", {emissive.x, emissive.y, emissive.z}},
							{"roughness", roughness},
							{"metalness", metalness},
							{"texture", texture ? texture : ""},
						}
					}
				}
			);
		};

		std::default_random_engine e;
		std::uniform_real_distribution<float> d(0.0f, 1.0f);

		addObject(jscene["objects"], HittableType::QUAD, vec3(), vec3(), vec3(20.0f), MaterialType::LAMBERT, vec3(1.0f), vec3(0.0f), 1.0f, 0.0f, nullptr);

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
					float roughness = d(e);
					HittableType type = static_cast<HittableType>(static_cast<uint32_t>(d(e) * 7.0f));
					addObject(jscene["objects"], type, center, vec3(), vec3(0.2f), MaterialType::LAMBERT_GGX, albedo, vec3(0.0f), roughness, metalness, nullptr);
				}
			}
		}

		addObject(jscene["objects"], HittableType::SPHERE, vec3(0.0f, 1.0f, 0.0f), vec3(), vec3(1.0f), MaterialType::LAMBERT_GGX, vec3(0.5f), vec3(0.0f), 0.5f, 0.0f, "earth.png");
	}

	std::ofstream infoFile("generated_scene.json", std::ios::out | std::ios::trunc);
	infoFile << std::setw(4) << jscene << std::endl;
	infoFile.close();
}

Camera loadScene(Pathtracer &pathtracer, const Params &params)
{
	// using a map lets us avoid loading the same texture multiple times
	std::map<std::string, uint32_t> texturePathToHandle;

	auto getTextureHandle = [&](const std::string &filepath) -> uint32_t
	{
		if (filepath.empty())
		{
			return 0;
		}

		// try to look for the texture in our map
		auto it = texturePathToHandle.find(filepath);
		if (it != texturePathToHandle.end())
		{
			return it->second;
		}
		// if this is the first time we encounter this texture, load it and store its handle in the map
		else
		{
			uint32_t handle = pathtracer.loadTexture(filepath.c_str());
			texturePathToHandle[filepath] = handle;
			return handle;
		}
	};

	// save extraction of values from json objects

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

	// open json file with scene
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

	// generate objects from scene file
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

			objects.push_back(CpuHittable(hittableType, position, vec3(radians(rotation.x), radians(rotation.y), radians(rotation.z)), scale, Material(materialType, baseColor, emissive, roughness, metalness, textureHandle)));
		}

		pathtracer.setScene(objects.size(), objects.data());
	}

	// skybox
	std::string skyboxTexturePath;
	if (getString(jscene, "skybox", skyboxTexturePath))
	{
		pathtracer.setSkyboxTextureHandle(getTextureHandle(skyboxTexturePath));
	}

	// camera
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

	return Camera(position, look_at, vec3(0.0f, 1.0f, 0.0f), radians(fovy), (float)params.m_width / params.m_height);
}