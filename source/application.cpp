#include "application.hpp"

#include "vulkan/vulkan_core.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

auto& context = graphics::internal::context;

// Типы сцены объявлены в namespace application, но используются общим кодом
// файла, поэтому импортируем их в анонимное пространство имён.
using application::Camera;
using application::Material;
using application::PointLight;
using application::SpotLight;

// ---------------------------------------------------------------------------
// Базовые типы
// ---------------------------------------------------------------------------

// Вершина: позиция и процедурно вычисленная нормаль грани.
struct Vertex {
	float position[3];
	float normal[3];
};

// Буфер, к памяти которого CPU имеет прямой доступ.
struct HostBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation{};
	void* mapped = nullptr;
	VkDeviceSize capacity = 0;
};

// Геометрия объекта: вершины хранятся подряд, отрисовка идёт без индексов.
struct Mesh {
	HostBuffer buffer;
	uint32_t vertexCount = 0;
};

// Данные сцены (набор дескрипторов 0, binding 0). Раскладка совпадает
// со структурой SceneBlock в шейдерах.
struct GpuSceneBlock {
	glm::mat4 viewProjection;
	glm::vec4 cameraPosition;
	glm::vec4 ambientColor;
	glm::vec4 directionalDirection;
	glm::vec4 directionalColor;
	glm::vec4 lightCounts;
};

// Данные модели (набор дескрипторов 1, binding 0).
struct GpuModelBlock {
	glm::mat4 model;
	glm::mat4 normalMatrix;
	glm::vec4 albedo;
	glm::vec4 specular;
};

// Элемент storage-буфера точечных источников.
struct GpuPointLight {
	glm::vec4 positionRange;
	glm::vec4 colorIntensity;
};

// Элемент storage-буфера прожекторных источников.
struct GpuSpotLight {
	glm::vec4 positionRange;
	glm::vec4 directionCosOuter;
	glm::vec4 colorIntensity;
	glm::vec4 innerCos;
};

// Раскладки структур должны совпадать с раскладкой в шейдерах.
static_assert(sizeof(GpuSceneBlock) == 144, "раскладка SceneBlock разошлась с шейдером");
static_assert(sizeof(GpuModelBlock) == 160, "раскладка ModelBlock разошлась с шейдером");
static_assert(sizeof(GpuPointLight) == 32, "раскладка PointLight разошлась с шейдером");
static_assert(sizeof(GpuSpotLight) == 64, "раскладка SpotLight разошлась с шейдером");

// ---------------------------------------------------------------------------
// Параметры сцены
// ---------------------------------------------------------------------------

// Число слотов под модели в общем uniform-буфере: каждый слот хранит
// матрицу трансформации и материал одного объекта.
constexpr uint32_t maxModelSlots = 128;

// Габариты сцены.
constexpr float floorHalfSize = 14.0f;
constexpr float spiralBaseHeight = 1.6f;
constexpr float spiralHalfHeight = 1.6f;
constexpr float maxOrbitRadius = 5.0f;
constexpr float minSpiralTurns = 0.5f;

constexpr float pi = std::numbers::pi_v<float>;

// Состояние сцены, изменяемое из интерфейса.
struct SceneState {
	// Рассеянный (ambient) свет.
	glm::vec3 ambientColor{0.14f, 0.15f, 0.2f};
	float ambientIntensity{1.0f};

	// Направленный источник: направление задано к источнику.
	glm::vec3 directionalColor{1.0f, 0.96f, 0.88f};
	float directionalIntensity{1.1f};
	float directionalYawDegrees{0.0f};
	float directionalPitchDegrees{55.0f};

	// Материалы объектов по модели Блинна-Фонга.
	Material pyramidMaterial;

	// Пол чуть темнее и холоднее, чтобы его освещение читалось иначе,
	// чем освещение пирамиды.
	Material floorMaterial{ glm::vec3{ 0.32f, 0.34f, 0.38f },
	                        glm::vec3{ 0.15f, 0.15f, 0.16f }, 24.0f };

	// Визуализация источников света.
	bool showLightVisualization{true};

	// Анимация объекта (наследие лабораторной №1).
	bool animationPaused = false;
	float animationTime = 0.0f;
	float animationSpeed = 1.0f;
	float orbitRadius = 1.5f;
	float spiralTurns = 2.0f;
	glm::vec3 objectPosition{0.0f, 0.0f, 0.0f};
	glm::vec3 objectRotationDegrees{0.0f, 0.0f, 0.0f};
	glm::vec3 objectScale{1.0f};
};

SceneState state;
Camera camera;
std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;

// ---------------------------------------------------------------------------
// Геометрия
// ---------------------------------------------------------------------------

// Нормаль грани вычисляется процедурно из положения вершин: при заданном
// ниже порядке обхода cross(v0 - v1, v2 - v1) смотрит наружу фигуры.
glm::vec3 faceNormal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
	return glm::normalize(glm::cross(v0 - v1, v2 - v1));
}

void pushTriangle(std::vector<Vertex>& out, const glm::vec3& a, const glm::vec3& b,
                  const glm::vec3& c) {
	const glm::vec3 normal = faceNormal(a, b, c);

	out.push_back(Vertex{ {a.x, a.y, a.z}, {normal.x, normal.y, normal.z} });
	out.push_back(Vertex{ {b.x, b.y, b.z}, {normal.x, normal.y, normal.z} });
	out.push_back(Vertex{ {c.x, c.y, c.z}, {normal.x, normal.y, normal.z} });
}

// Квад: origin — угол, u и v — стороны. Наружная нормаль получается, если
// cross(u, v) направлен внутрь фигуры.
void pushQuad(std::vector<Vertex>& out, const glm::vec3& origin, const glm::vec3& u,
              const glm::vec3& v) {
	const glm::vec3 p0 = origin;
	const glm::vec3 p1 = origin + u;
	const glm::vec3 p2 = origin + u + v;
	const glm::vec3 p3 = origin + v;

	pushTriangle(out, p0, p1, p2);
	pushTriangle(out, p0, p2, p3);
}

// Пирамида из лабораторной №1: четыре боковые грани и дно. Порядок обхода
// вершин совпадает с индексным буфером из лабораторной №1.
std::vector<Vertex> makePyramid(float halfSize, float height) {
	const glm::vec3 v0{ halfSize, -halfSize, halfSize };
	const glm::vec3 v1{ halfSize, -halfSize, -halfSize };
	const glm::vec3 v2{ -halfSize, -halfSize, -halfSize };
	const glm::vec3 v3{ -halfSize, -halfSize, halfSize };
	const glm::vec3 v4{ 0.0f, height - halfSize, 0.0f };

	std::vector<Vertex> vertices;
	vertices.reserve(15);

	pushTriangle(vertices, v4, v0, v3);
	pushTriangle(vertices, v4, v1, v0);
	pushTriangle(vertices, v4, v2, v1);
	pushTriangle(vertices, v4, v3, v2);
	pushTriangle(vertices, v1, v2, v0);
	pushTriangle(vertices, v2, v3, v0);

	return vertices;
}

// Пол: квад в плоскости XZ с нормалью вверх.
std::vector<Vertex> makeFloor(float halfSize) {
	std::vector<Vertex> vertices;
	vertices.reserve(6);

	pushQuad(vertices, glm::vec3{ -halfSize, 0.0f, -halfSize },
	         glm::vec3{ 2.0f * halfSize, 0.0f, 0.0f },
	         glm::vec3{ 0.0f, 0.0f, 2.0f * halfSize });

	return vertices;
}

// Куб для визуализации источников света: единичный, центр в начале координат.
std::vector<Vertex> makeCube() {
	constexpr float h = 0.5f;

	std::vector<Vertex> vertices;
	vertices.reserve(36);

	// +X
	pushQuad(vertices, { h, -h, -h }, { 0.0f, 0.0f, 2.0f * h }, { 0.0f, 2.0f * h, 0.0f });
	// -X
	pushQuad(vertices, { -h, -h, -h }, { 0.0f, 2.0f * h, 0.0f }, { 0.0f, 0.0f, 2.0f * h });
	// +Y
	pushQuad(vertices, { -h, h, -h }, { 2.0f * h, 0.0f, 0.0f }, { 0.0f, 0.0f, 2.0f * h });
	// -Y
	pushQuad(vertices, { -h, -h, -h }, { 0.0f, 0.0f, 2.0f * h }, { 2.0f * h, 0.0f, 0.0f });
	// +Z
	pushQuad(vertices, { -h, -h, h }, { 0.0f, 2.0f * h, 0.0f }, { 2.0f * h, 0.0f, 0.0f });
	// -Z
	pushQuad(vertices, { -h, -h, -h }, { 2.0f * h, 0.0f, 0.0f }, { 0.0f, 2.0f * h, 0.0f });

	return vertices;
}

// Конус с вершиной в начале координат и осью вдоль +Z: используется для
// визуализации направления и угла прожекторного источника.
std::vector<Vertex> makeCone(int segments) {
	std::vector<Vertex> vertices;
	vertices.reserve(static_cast<size_t>(segments) * 6);

	for (int i = 0; i < segments; ++i) {
		const float a0 = 2.0f * pi * float(i) / float(segments);
		const float a1 = 2.0f * pi * float(i + 1) / float(segments);

		const glm::vec3 p0{ std::cos(a0), std::sin(a0), 1.0f };
		const glm::vec3 p1{ std::cos(a1), std::sin(a1), 1.0f };

		// Боковая поверхность: нормаль смотрит наружу и назад к вершине.
		pushTriangle(vertices, { 0.0f, 0.0f, 0.0f }, p0, p1);
		// Основание: нормаль смотрит вдоль +Z.
		pushTriangle(vertices, { 0.0f, 0.0f, 1.0f }, p1, p0);
	}

	return vertices;
}

// ---------------------------------------------------------------------------
// Ресурсы Vulkan
// ---------------------------------------------------------------------------

Mesh pyramidMesh;
Mesh floorMesh;
Mesh cubeMesh;
Mesh coneMesh;

VkDescriptorPool descriptorPool;
VkDescriptorSetLayout sceneSetLayout;
VkDescriptorSetLayout modelSetLayout;
VkDescriptorSet sceneDescriptorSet;
VkDescriptorSet modelDescriptorSet;

VkPipelineLayout pipelineLayout;
VkPipeline scenePipeline;
VkPipeline unlitPipeline;
VkPipeline unlitBlendPipeline;

HostBuffer sceneUniformBuffer;
HostBuffer modelUniformBuffer;
HostBuffer pointLightBuffer;
HostBuffer spotLightBuffer;

// Шаг между слотами общего буфера моделей. Динамическое смещение
// дескриптора должно быть кратно ограничению драйвера, поэтому шаг
// вычисляется по нему в createDescriptorData.
VkDeviceSize modelSlotStride = 256;

// ---------------------------------------------------------------------------
// Вспомогательные функции
// ---------------------------------------------------------------------------

std::vector<char> readFile(const std::string& path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file) {
		std::cerr << "Failed to open file: " << path << '\n';
		return {};
	}

	const size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

	return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& shaderCode) {
	if (shaderCode.empty()) {
		return VK_NULL_HANDLE;
	}

	VkShaderModuleCreateInfo shaderModuleCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = shaderCode.size(),
		.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
	};

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &shaderModuleCreateInfo, nullptr,
	                         &shaderModule) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan shader module\n";
		return VK_NULL_HANDLE;
	}

	return shaderModule;
}

// Буфер, доступный CPU: память выделяется VMA (HOST_VISIBLE | HOST_COHERENT),
// данные копируются в отображённую память.
bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data,
                      HostBuffer& out) {
	const VkBufferCreateInfo bufferInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};

	const VmaAllocationCreateInfo allocInfo{
		.usage = VMA_MEMORY_USAGE_CPU_ONLY
	};

	if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocInfo, &out.buffer,
	                    &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate buffer\n";
		return false;
	}

	out.capacity = size;

	if (vmaMapMemory(context.allocator, out.allocation, &out.mapped) != VK_SUCCESS) {
		std::cerr << "Failed to map buffer memory\n";
		vmaDestroyBuffer(context.allocator, out.buffer, out.allocation);
		out.buffer = VK_NULL_HANDLE;
		out.allocation = VmaAllocation{};
		out.capacity = 0;
		return false;
	}

	// Память отображена постоянно: uniform и storage буферы обновляются
	// каждый кадр, а HOST_COHERENT не требует ручного flush.
	std::memset(out.mapped, 0, static_cast<size_t>(size));

	if (data != nullptr) {
		std::memcpy(out.mapped, data, static_cast<size_t>(size));
	}

	return true;
}

void destroyHostBuffer(HostBuffer& buffer) {
	if (buffer.buffer == VK_NULL_HANDLE) {
		return;
	}

	if (buffer.mapped != nullptr) {
		vmaUnmapMemory(context.allocator, buffer.allocation);
		buffer.mapped = nullptr;
	}

	vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
	buffer.buffer = VK_NULL_HANDLE;
	buffer.allocation = VmaAllocation{};
	buffer.capacity = 0;
}

bool createMesh(Mesh& mesh, const std::vector<Vertex>& vertices) {
	if (vertices.empty()) {
		return false;
	}

	if (!createHostBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                      vertices.data(), mesh.buffer)) {
		return false;
	}

	mesh.vertexCount = static_cast<uint32_t>(vertices.size());

	return true;
}

// ---------------------------------------------------------------------------
// Траектория объекта
// ---------------------------------------------------------------------------

// Точка траектории для прогресса p в диапазоне [0, 1].
glm::vec3 trajectoryPoint(float progress) {
	const float angle = progress * 2.0f * pi * state.spiralTurns;

	return {
		std::cos(angle) * state.orbitRadius,
		spiralBaseHeight + std::sin(progress * pi) * spiralHalfHeight,
		std::sin(angle) * state.orbitRadius,
	};
}

// Касательная к траектории — производная точки по прогрессу, без нормировки.
glm::vec3 trajectoryTangent(float progress) {
	const float angle = progress * 2.0f * pi * state.spiralTurns;
	const float angularSpeed = 2.0f * pi * state.spiralTurns;

	return {
		-std::sin(angle) * state.orbitRadius * angularSpeed,
		std::cos(progress * pi) * pi * spiralHalfHeight,
		std::cos(angle) * state.orbitRadius * angularSpeed,
	};
}

// За цикл объект обходит окружность ровно spiralTurns раз, поэтому длина
// пути постоянна и скорость не зависит от радиуса и числа витков.
float cycleDuration() {
	const float pathLength = 2.0f * pi * state.orbitRadius * state.spiralTurns;
	const float speed = std::max(state.animationSpeed, 0.01f);

	return std::max(pathLength / speed, 0.05f);
}

// Матрица трансформации главного объекта: движение по траектории плюс
// пользовательский поворот и масштаб.
glm::mat4 pyramidModelMatrix() {
	const float progress = state.animationTime / cycleDuration();

	const glm::vec3 orbitOffset = trajectoryPoint(progress);
	const glm::vec3 tangent = glm::normalize(trajectoryTangent(progress));

	// Горизонтальное направление от объекта к оси траектории.
	glm::vec3 inward = glm::vec3(
		state.objectPosition.x - orbitOffset.x,
		0.0f,
		state.objectPosition.z - orbitOffset.z
	);

	if (glm::dot(inward, inward) < 1e-8f) {
		inward = glm::vec3(0.0f, 0.0f, -1.0f);
	} else {
		inward = glm::normalize(inward);
	}

	const glm::vec3 side = glm::normalize(glm::cross(tangent, inward));

	// Локальная +Y смотрит вдоль движения, диагональ (X+Z)/√2 — на ось.
	// Определитель матрицы равен +1, поэтому отсечение граней не инвертируется.
	constexpr float invSqrt2 = 0.70710678118654752f;

	const glm::mat4 orientation{
		glm::vec4((inward + side) * invSqrt2, 0.0f),
		glm::vec4(tangent, 0.0f),
		glm::vec4((inward - side) * invSqrt2, 0.0f),
		glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
	};

	glm::mat4 local = glm::mat4(1.0f);
	local = glm::scale(local, state.objectScale);
	local = glm::rotate(local, glm::radians(state.objectRotationDegrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
	local = glm::rotate(local, glm::radians(state.objectRotationDegrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
	local = glm::rotate(local, glm::radians(state.objectRotationDegrees.z), glm::vec3(0.0f, 0.0f, 1.0f));

	return glm::translate(glm::mat4(1.0f), state.objectPosition + orbitOffset) * orientation * local;
}

// ---------------------------------------------------------------------------
// Направления источников света
// ---------------------------------------------------------------------------

// Направление по азимуту и тангажу: азимут 0 смотрит вдоль -Z, положительный
// тангаж поднимает направление вверх.
glm::vec3 directionFromAngles(float yawDegrees, float pitchDegrees) {
	const float yaw = glm::radians(yawDegrees);
	const float pitch = glm::radians(pitchDegrees);

	return glm::normalize(glm::vec3{
		std::sin(yaw) * std::cos(pitch),
		std::sin(pitch),
		-std::cos(yaw) * std::cos(pitch)
	});
}

// ---------------------------------------------------------------------------
// Пайплайны
// ---------------------------------------------------------------------------

const VkVertexInputBindingDescription vertexBindingDescription{
	.binding = 0,
	.stride = sizeof(Vertex),
	.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
};

const VkVertexInputAttributeDescription vertexAttributeDescriptions[] = {
	{
		.location = 0,
		.binding = 0,
		.format = VK_FORMAT_R32G32B32_SFLOAT,
		.offset = offsetof(Vertex, position)
	},
	{
		.location = 1,
		.binding = 0,
		.format = VK_FORMAT_R32G32B32_SFLOAT,
		.offset = offsetof(Vertex, normal)
	},
};

bool createDescriptorLayouts() {
	const VkDescriptorSetLayoutBinding sceneBindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
		},
	};

	const VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = static_cast<uint32_t>(std::size(sceneBindings)),
		.pBindings = sceneBindings
	};

	if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr,
	                                &sceneSetLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout for the scene\n";
		return false;
	}

	// Набор дескрипторов модели: данные о трансформации и материале одного
	// объекта. Динамическое смещение позволяет менять объект, не пересоздавая
	// набор дескрипторов.
	const VkDescriptorSetLayoutBinding modelBinding{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
	};

	const VkDescriptorSetLayoutCreateInfo modelLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &modelBinding
	};

	if (vkCreateDescriptorSetLayout(context.device, &modelLayoutInfo, nullptr,
	                                &modelSetLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout for the model\n";
		return false;
	}

	return true;
}

bool createDescriptorPoolAndSets() {
	const VkDescriptorPoolSize poolSizes[] = {
		// Набор дескрипторов сцены: uniform-буфер параметров сцены.
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1
		},
		// Набор дескрипторов модели: uniform-буфер с динамическим смещением.
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2
		},
	};

	const VkDescriptorPoolCreateInfo poolInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes)),
		.pPoolSizes = poolSizes
	};

	if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = { sceneSetLayout, modelSetLayout };

	const VkDescriptorSetAllocateInfo setInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = 2,
		.pSetLayouts = layouts
	};

	VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

	if (vkAllocateDescriptorSets(context.device, &setInfo, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	sceneDescriptorSet = sets[0];
	modelDescriptorSet = sets[1];

	return true;
}

// Запись дескриптора storage-буфера. Вызывается при пересоздании буфера.
void writeStorageDescriptor(uint32_t binding, const HostBuffer& buffer) {
	if (buffer.buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkDescriptorBufferInfo bufferInfo{
		.buffer = buffer.buffer,
		.offset = 0,
		.range = buffer.capacity
	};

	const VkWriteDescriptorSet write{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = sceneDescriptorSet,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &bufferInfo
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

bool createDescriptorData() {
	// Шаг слота общего буфера моделей должен быть кратен ограничению драйвера
	// на смещение динамического uniform-буфера.
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);

	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	modelSlotStride = ((sizeof(GpuModelBlock) + alignment - 1) / alignment) * alignment;

	// Uniform-буфер сцены.
	if (!createHostBuffer(sizeof(GpuSceneBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                      nullptr, sceneUniformBuffer)) {
		return false;
	}

	// Общий буфер моделей: каждая модель занимает свой слот.
	if (!createHostBuffer(maxModelSlots * modelSlotStride,
	                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
	                      modelUniformBuffer)) {
		return false;
	}

	// Storage-буферы источников света создаются с минимальным размером,
	// чтобы дескриптор всегда ссылался на валидный буфер.
	if (!createHostBuffer(sizeof(GpuPointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                      nullptr, pointLightBuffer)) {
		return false;
	}

	if (!createHostBuffer(sizeof(GpuSpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                      nullptr, spotLightBuffer)) {
		return false;
	}

	const VkDescriptorBufferInfo sceneUniformInfo{
		.buffer = sceneUniformBuffer.buffer,
		.offset = 0,
		.range = sizeof(GpuSceneBlock)
	};

	// Диапазон динамического дескриптора — размер данных одного объекта.
	// Смещение добавляется к нему, поэтому диапазон не может быть больше
	// размера блока, иначе последний слот вышел бы за границы буфера.
	const VkDescriptorBufferInfo modelUniformInfo{
		.buffer = modelUniformBuffer.buffer,
		.offset = 0,
		.range = sizeof(GpuModelBlock)
	};

	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = sceneDescriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &sceneUniformInfo
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = modelDescriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &modelUniformInfo
		},
	};

	vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(std::size(writes)), writes,
	                        0, nullptr);

	writeStorageDescriptor(1, pointLightBuffer);
	writeStorageDescriptor(2, spotLightBuffer);

	return true;
}

bool createPipelineLayout() {
	const VkDescriptorSetLayout layouts[] = { sceneSetLayout, modelSetLayout };

	const VkPipelineLayoutCreateInfo layoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = layouts
	};

	if (vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	return true;
}

// Создание графического пайплайна. Общая часть: одинаковые состояния
// вершинных данных, отсечения, глубины и динамических областей.
bool createGraphicsPipeline(VkShaderModule vertexModule, VkShaderModule fragmentModule,
                            VkPipeline& pipeline) {
	if (vertexModule == VK_NULL_HANDLE || fragmentModule == VK_NULL_HANDLE) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertexModule,
			.pName = "main"
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragmentModule,
			.pName = "main"
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertexInput{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertexBindingDescription,
		.vertexAttributeDescriptionCount =
			static_cast<uint32_t>(std::size(vertexAttributeDescriptions)),
		.pVertexAttributeDescriptions = vertexAttributeDescriptions
	};

	const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	const VkPipelineViewportStateCreateInfo viewportState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1
	};

	const VkPipelineRasterizationStateCreateInfo rasterization{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		// Ось Y отражается в матрице проекции, поэтому winding инвертирован.
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};

	const VkPipelineMultisampleStateCreateInfo multisample{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE
	};

	const VkPipelineDepthStencilStateCreateInfo depthStencil{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS
	};

	const VkPipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
	};

	const VkPipelineColorBlendStateCreateInfo colorBlend{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	const std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	const VkPipelineDynamicStateCreateInfo dynamicState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};

	const VkGraphicsPipelineCreateInfo pipelineInfo{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertexInput,
		.pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlend,
		.pDynamicState = &dynamicState,
		.layout = pipelineLayout,
		.renderPass = context.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
	                              &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

// Пайплайн прозрачной геометрии визуализации: постоянный цвет, альфа
// смешивание, запись в буфер глубины отключена, задние грани не отсекаются,
// чтобы объём конуса читался целиком.
bool createBlendPipeline(VkShaderModule vertexModule, VkShaderModule fragmentModule) {
	if (vertexModule == VK_NULL_HANDLE || fragmentModule == VK_NULL_HANDLE) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertexModule,
			.pName = "main"
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragmentModule,
			.pName = "main"
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertexInput{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertexBindingDescription,
		.vertexAttributeDescriptionCount =
			static_cast<uint32_t>(std::size(vertexAttributeDescriptions)),
		.pVertexAttributeDescriptions = vertexAttributeDescriptions
	};

	const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	const VkPipelineViewportStateCreateInfo viewportState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1
	};

	const VkPipelineRasterizationStateCreateInfo rasterization{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		// Задние грани отсекаются: иначе каждая грань конуса рисуется
		// дважды (ближняя и дальняя поверхность), прозрачность складывается
		// в несколько слоёв, и конус выглядит плотным пятном.
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};

	const VkPipelineMultisampleStateCreateInfo multisample{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE
	};

	const VkPipelineDepthStencilStateCreateInfo depthStencil{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_LESS
	};

	const VkPipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable = VK_TRUE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
	};

	const VkPipelineColorBlendStateCreateInfo colorBlend{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	const std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	const VkPipelineDynamicStateCreateInfo dynamicState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};

	const VkGraphicsPipelineCreateInfo pipelineInfo{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertexInput,
		.pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlend,
		.pDynamicState = &dynamicState,
		.layout = pipelineLayout,
		.renderPass = context.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
	                              &unlitBlendPipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan blending pipeline\n";
		return false;
	}

	return true;
}

bool createPipelines() {
	const auto vertexCode = readFile("shaders/scene.vert.spv");
	const auto fragmentCode = readFile("shaders/scene.frag.spv");

	const VkShaderModule vertexModule = createShaderModule(vertexCode);
	const VkShaderModule fragmentModule = createShaderModule(fragmentCode);

	const bool sceneOk = createGraphicsPipeline(vertexModule, fragmentModule, scenePipeline);

	vkDestroyShaderModule(context.device, vertexModule, nullptr);
	vkDestroyShaderModule(context.device, fragmentModule, nullptr);

	if (!sceneOk) {
		return false;
	}

	const auto unlitVertexCode = readFile("shaders/unlit.vert.spv");
	const auto unlitFragmentCode = readFile("shaders/unlit.frag.spv");

	const VkShaderModule unlitVertexModule = createShaderModule(unlitVertexCode);
	const VkShaderModule unlitFragmentModule = createShaderModule(unlitFragmentCode);

	const bool unlitOk = createGraphicsPipeline(unlitVertexModule, unlitFragmentModule,
	                                            unlitPipeline);
	const bool unlitBlendOk = createBlendPipeline(unlitVertexModule, unlitFragmentModule);

	vkDestroyShaderModule(context.device, unlitVertexModule, nullptr);
	vkDestroyShaderModule(context.device, unlitFragmentModule, nullptr);

	return unlitOk && unlitBlendOk;
}

// ---------------------------------------------------------------------------
// Данные источников света
// ---------------------------------------------------------------------------

// Пересоздание storage-буфера, если выросло число источников. Буфер
// освобождается и создаётся заново, а дескриптор обновляется.
bool ensureStorageCapacity(HostBuffer& buffer, uint32_t binding, VkDeviceSize required) {
	if (buffer.buffer != VK_NULL_HANDLE && buffer.capacity >= required) {
		return true;
	}

	const VkDeviceSize newSize = std::max(required, buffer.capacity * 2);

	destroyHostBuffer(buffer);

	if (!createHostBuffer(newSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, buffer)) {
		return false;
	}

	writeStorageDescriptor(binding, buffer);

	return true;
}

// ---------------------------------------------------------------------------
// Список отрисовки
// ---------------------------------------------------------------------------

struct DrawItem {
	const Mesh* mesh = nullptr;
	glm::mat4 model = glm::mat4(1.0f);
	glm::vec4 albedo{1.0f, 1.0f, 1.0f, 1.0f};
	glm::vec4 specular{0.5f, 0.5f, 0.5f, 32.0f};
};

// Освещённые объекты, фигуры визуализации источников и прозрачные конусы
// рисуются разными пайплайнами, поэтому хранятся в отдельных списках.
std::vector<DrawItem> litItems;
std::vector<DrawItem> unlitItems;
std::vector<DrawItem> transparentItems;

} // namespace

// ---------------------------------------------------------------------------
// Публичный интерфейс приложения
// ---------------------------------------------------------------------------

namespace application {

namespace {

GLFWwindow* window = nullptr;

bool mouseButtonHeld = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

// При 90° ось поворота совпадает с направлением взгляда.
constexpr float maxPitchDegrees = 89.0f;
constexpr double mouseSensitivity = 0.3;
constexpr float cameraMoveSpeed = 5.0f;

} // namespace

// Запоминаем окно и подписываемся на события мыши и колеса.
void attachWindow(GLFWwindow* const glfwWindow) {
	window = glfwWindow;

	glfwSetMouseButtonCallback(window, [](GLFWwindow*, int button, int action, int) {
		application::onMouseButton(button, action);
	});

	glfwSetCursorPosCallback(window, [](GLFWwindow*, double xpos, double ypos) {
		application::onCursorPos(xpos, ypos);
	});

	glfwSetScrollCallback(window, [](GLFWwindow*, double, double yoffset) {
		application::onScroll(yoffset);
	});
}

// Вращение камеры по зажатой левой кнопке мыши.
void onMouseButton(int button, int action) {
	if (button != GLFW_MOUSE_BUTTON_LEFT) {
		return;
	}

	mouseButtonHeld = (action == GLFW_PRESS);

	if (mouseButtonHeld && window != nullptr) {
		glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
	}
}

// Перетаскивание курсора поворачивает камеру. Пока курсор над интерфейсом
// ImGui, поворот не выполняется, но позиция запоминается.
void onCursorPos(double xpos, double ypos) {
	if (!mouseButtonHeld) {
		return;
	}

	if (ImGui::GetIO().WantCaptureMouse) {
		lastMouseX = xpos;
		lastMouseY = ypos;
		return;
	}

	const double dx = xpos - lastMouseX;
	const double dy = ypos - lastMouseY;

	lastMouseX = xpos;
	lastMouseY = ypos;

	camera.yawDegrees += static_cast<float>(dx * mouseSensitivity);
	camera.pitchDegrees = glm::clamp(
		camera.pitchDegrees - static_cast<float>(dy * mouseSensitivity),
		-maxPitchDegrees, maxPitchDegrees
	);
}

// Колесо мыши меняет угол обзора камеры.
void onScroll(double yoffset) {
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}

	camera.fovDegrees = glm::clamp(
		camera.fovDegrees - static_cast<float>(yoffset) * 3.0f, 15.0f, 90.0f
	);
}

bool initialize() {
	// При любой ошибке освобождаем уже созданные ресурсы: иначе их уничтожит
	// освобождение аллокатора, что приводит к assertion в VMA.
	auto fail = []() {
		shutdown();
		return false;
	};

	// Геометрия сцены.
	if (!createMesh(pyramidMesh, makePyramid(0.5f, 1.0f))) {
		return fail();
	}

	if (!createMesh(floorMesh, makeFloor(floorHalfSize))) {
		return fail();
	}

	if (!createMesh(cubeMesh, makeCube())) {
		return fail();
	}

	if (!createMesh(coneMesh, makeCone(24))) {
		return fail();
	}

	if (!createDescriptorLayouts()) {
		return fail();
	}

	if (!createDescriptorPoolAndSets()) {
		return fail();
	}

	if (!createDescriptorData()) {
		return fail();
	}

	if (!createPipelineLayout()) {
		return fail();
	}

	if (!createPipelines()) {
		return fail();
	}

	// Стартовый набор источников света.
	PointLight warmPoint;
	warmPoint.position = glm::vec3{ 2.6f, 2.4f, 2.2f };
	warmPoint.color = glm::vec3{ 1.0f, 0.78f, 0.45f };
	warmPoint.intensity = 22.0f;
	warmPoint.range = 12.0f;
	pointLights.push_back(warmPoint);

	SpotLight overheadSpot;
	overheadSpot.position = glm::vec3{ -2.0f, 5.0f, -1.0f };
	overheadSpot.color = glm::vec3{ 0.7f, 0.82f, 1.0f };
	overheadSpot.intensity = 90.0f;
	overheadSpot.range = 18.0f;
	overheadSpot.yawDegrees = 25.0f;
	overheadSpot.pitchDegrees = -70.0f;
	overheadSpot.outerAngleDegrees = 34.0f;
	overheadSpot.innerAngleDegrees = 15.0f;
	spotLights.push_back(overheadSpot);

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, scenePipeline, nullptr);
	vkDestroyPipeline(context.device, unlitPipeline, nullptr);
	vkDestroyPipeline(context.device, unlitBlendPipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

	destroyHostBuffer(pointLightBuffer);
	destroyHostBuffer(spotLightBuffer);
	destroyHostBuffer(modelUniformBuffer);
	destroyHostBuffer(sceneUniformBuffer);

	destroyHostBuffer(pyramidMesh.buffer);
	destroyHostBuffer(floorMesh.buffer);
	destroyHostBuffer(cubeMesh.buffer);
	destroyHostBuffer(coneMesh.buffer);

	if (descriptorPool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
		descriptorPool = VK_NULL_HANDLE;
	}

	if (sceneSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
		sceneSetLayout = VK_NULL_HANDLE;
	}

	if (modelSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, modelSetLayout, nullptr);
		modelSetLayout = VK_NULL_HANDLE;
	}
}

// ---------------------------------------------------------------------------
// Интерфейс
// ---------------------------------------------------------------------------

namespace {

void drawCameraPanel() {
	ImGui::Begin("Камера");

	ImGui::DragFloat3("Позиция", &camera.position.x, 0.05f);
	ImGui::DragFloat("Азимут", &camera.yawDegrees, 0.5f, -360.0f, 360.0f, "%.0f grad");
	ImGui::DragFloat("Тангаж", &camera.pitchDegrees, 0.5f, -maxPitchDegrees, maxPitchDegrees,
	                 "%.0f grad");
	ImGui::DragFloat("Угол обзора", &camera.fovDegrees, 0.5f, 15.0f, 90.0f, "%.0f grad");

	ImGui::Separator();
	ImGui::TextUnformatted("ЛКМ + движение мыши — поворот камеры");
	ImGui::TextUnformatted("Колесо мыши — угол обзора");
	ImGui::TextUnformatted("W / A / S / D — перемещение, Q / E — вверх и вниз");

	ImGui::End();
}

void drawMaterialControls(const char* id, Material& material) {
	ImGui::PushID(id);

	ImGui::ColorEdit3("Альбедо", &material.albedo.x);
	ImGui::ColorEdit3("Цвет блика", &material.specularColor.x);
	ImGui::DragFloat("Коэффициент блика", &material.shininess, 1.0f, 1.0f, 256.0f, "%.0f");

	ImGui::PopID();
}

void drawScenePanel() {
	ImGui::Begin("Сцена");

	ImGui::TextUnformatted("Рассеянный свет (ambient)");
	ImGui::ColorEdit3("Цвет ambient", &state.ambientColor.x);
	ImGui::DragFloat("Сила ambient", &state.ambientIntensity, 0.01f, 0.0f, 5.0f, "%.2f");

	ImGui::Separator();
	ImGui::TextUnformatted("Направленный источник");
	ImGui::ColorEdit3("Цвет направленного", &state.directionalColor.x);
	ImGui::DragFloat("Интенсивность направленного", &state.directionalIntensity, 0.01f, 0.0f,
	                 10.0f, "%.2f");
	ImGui::DragFloat("Азимут к источнику", &state.directionalYawDegrees, 0.5f, -360.0f, 360.0f,
	                 "%.0f grad");
	ImGui::DragFloat("Тангаж к источнику", &state.directionalPitchDegrees, 0.5f, -89.0f, 89.0f,
	                 "%.0f grad");

	ImGui::Separator();
	ImGui::TextUnformatted("Материал пирамиды");
	drawMaterialControls("pyramid", state.pyramidMaterial);

	ImGui::Separator();
	ImGui::TextUnformatted("Материал пола");
	drawMaterialControls("floor", state.floorMaterial);

	ImGui::Separator();
	ImGui::TextUnformatted("Объект");
	ImGui::DragFloat3("Позиция объекта", &state.objectPosition.x, 0.05f);
	ImGui::DragFloat3("Поворот объекта", &state.objectRotationDegrees.x, 0.5f, -360.0f, 360.0f);
	ImGui::DragFloat3("Масштаб объекта", &state.objectScale.x, 0.05f, 0.1f, 5.0f);

	ImGui::Separator();
	ImGui::TextUnformatted("Анимация");
	if (ImGui::Button(state.animationPaused ? "Продолжить" : "Пауза")) {
		state.animationPaused = !state.animationPaused;
	}
	ImGui::SameLine();
	if (ImGui::Button("Сброс анимации")) {
		state.animationTime = 0.0f;
		state.animationPaused = false;
	}

	// Перемотка таймлайна останавливает анимацию, иначе значение ползунка
	// тут же сдвинулось бы на следующий кадр.
	if (ImGui::SliderFloat("Таймлайн", &state.animationTime, 0.0f, cycleDuration(), "%.2f s")) {
		state.animationPaused = true;
	}

	ImGui::SliderFloat("Скорость", &state.animationSpeed, 0.1f, 5.0f, "%.2fx");
	ImGui::SliderFloat("Радиус спирали", &state.orbitRadius, 0.2f, maxOrbitRadius, "%.2f");
	ImGui::SliderFloat("Плотность витков", &state.spiralTurns, minSpiralTurns, 6.0f, "%.1f");

	ImGui::End();
}

void drawLightPanels() {
	ImGui::Begin("Источники света");

	ImGui::Checkbox("Показывать визуализацию источников", &state.showLightVisualization);

	ImGui::Separator();
	ImGui::TextUnformatted("Точечные источники");

	int removePointLight = -1;

	for (size_t i = 0; i < pointLights.size(); ++i) {
		PointLight& light = pointLights[i];

		ImGui::PushID(static_cast<int>(i));

		const std::string header = "Точечный #" + std::to_string(i);
		if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Включён", &light.enabled);
			ImGui::DragFloat3("Позиция", &light.position.x, 0.05f);
			ImGui::ColorEdit3("Цвет", &light.color.x);
			ImGui::DragFloat("Интенсивность", &light.intensity, 0.1f, 0.0f, 200.0f, "%.1f");
			ImGui::DragFloat("Радиус действия", &light.range, 0.1f, 0.5f, 60.0f, "%.1f");
			ImGui::Checkbox("Прикрепить к пирамиде", &light.attachedToObject);

			if (ImGui::Button("Удалить")) {
				removePointLight = static_cast<int>(i);
			}
		}

		ImGui::PopID();
	}

	if (ImGui::Button("Добавить точечный источник")) {
		PointLight light;
		light.position = glm::vec3{ 0.0f, 2.0f, 0.0f };
		light.color = glm::vec3{ 0.6f, 0.85f, 1.0f };
		light.intensity = 15.0f;
		light.range = 10.0f;
		pointLights.push_back(light);
	}

	if (removePointLight >= 0) {
		pointLights.erase(pointLights.begin() + removePointLight);
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Прожекторные источники");

	int removeSpotLight = -1;

	for (size_t i = 0; i < spotLights.size(); ++i) {
		SpotLight& light = spotLights[i];

		ImGui::PushID(1000 + static_cast<int>(i));

		const std::string header = "Прожекторный #" + std::to_string(i);
		if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Включён", &light.enabled);
			ImGui::DragFloat3("Позиция", &light.position.x, 0.05f);
			ImGui::ColorEdit3("Цвет", &light.color.x);
			ImGui::DragFloat("Интенсивность", &light.intensity, 0.5f, 0.0f, 500.0f, "%.1f");
			ImGui::DragFloat("Радиус действия", &light.range, 0.1f, 1.0f, 60.0f, "%.1f");
			ImGui::DragFloat("Азимут", &light.yawDegrees, 0.5f, -360.0f, 360.0f, "%.0f grad");
			ImGui::DragFloat("Тангаж", &light.pitchDegrees, 0.5f, -89.0f, 89.0f, "%.0f grad");
			ImGui::DragFloat("Внешний угол", &light.outerAngleDegrees, 0.5f, 1.0f, 89.0f,
			                 "%.0f grad");
			ImGui::DragFloat("Внутренний угол", &light.innerAngleDegrees, 0.5f, 0.5f, 89.0f,
			                 "%.0f grad");
			ImGui::Checkbox("Прикрепить к пирамиде", &light.attachedToObject);

			// Мягкий край возможен, только если внешний угол шире внутреннего.
			if (light.innerAngleDegrees > light.outerAngleDegrees) {
				light.innerAngleDegrees = light.outerAngleDegrees;
			}

			if (ImGui::Button("Удалить")) {
				removeSpotLight = static_cast<int>(i);
			}
		}

		ImGui::PopID();
	}

	if (ImGui::Button("Добавить прожекторный источник")) {
		SpotLight light;
		light.position = glm::vec3{ 0.0f, 4.0f, 0.0f };
		light.color = glm::vec3{ 1.0f, 1.0f, 1.0f };
		light.intensity = 60.0f;
		light.range = 15.0f;
		light.pitchDegrees = -70.0f;
		light.outerAngleDegrees = 30.0f;
		light.innerAngleDegrees = 12.0f;
		spotLights.push_back(light);
	}

	if (removeSpotLight >= 0) {
		spotLights.erase(spotLights.begin() + removeSpotLight);
	}

	ImGui::End();
}

// Перемещение камеры клавишами WASD.
void updateCameraMovement(double deltaTime) {
	if (window == nullptr || ImGui::GetIO().WantCaptureKeyboard) {
		return;
	}

	const float speed = cameraMoveSpeed * static_cast<float>(deltaTime);

	const float yaw = glm::radians(camera.yawDegrees);
	const float pitch = glm::radians(camera.pitchDegrees);

	const glm::vec3 forward{ std::sin(yaw) * std::cos(pitch), std::sin(pitch),
		                     -std::cos(yaw) * std::cos(pitch) };
	const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));

	glm::vec3 offset(0.0f);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		offset += forward;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		offset -= forward;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		offset += right;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		offset -= right;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		offset.y -= 1.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		offset.y += 1.0f;
	}

	if (glm::dot(offset, offset) > 0.0f) {
		camera.position += glm::normalize(offset) * speed;
	}
}

} // namespace

void update(double time) {
	static double previousUpdateTime = 0.0;
	static bool hasPreviousUpdateTime = false;

	if (!hasPreviousUpdateTime) {
		hasPreviousUpdateTime = true;
		previousUpdateTime = time;
	}

	const double deltaTime = time - previousUpdateTime;
	previousUpdateTime = time;

	if (!state.animationPaused) {
		state.animationTime += static_cast<float>(deltaTime);
	}

	// После смены скорости длительность цикла меняется, и время может
	// выйти за её новые границы.
	const float duration = cycleDuration();
	state.animationTime -= std::floor(state.animationTime / duration) * duration;
	state.animationTime = std::clamp(state.animationTime, 0.0f, duration);

	updateCameraMovement(deltaTime);

	drawCameraPanel();
	drawScenePanel();
	drawLightPanels();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	// Матрицы вида и проекции.
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	const float yaw = glm::radians(camera.yawDegrees);
	const float pitch = glm::radians(camera.pitchDegrees);

	const glm::vec3 forward{ std::sin(yaw) * std::cos(pitch), std::sin(pitch),
		                     -std::cos(yaw) * std::cos(pitch) };
	const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
	const glm::vec3 up = glm::cross(right, forward);

	const glm::mat4 view = glm::lookAt(camera.position, camera.position + forward, up);

	glm::mat4 projection = glm::perspective(
		glm::radians(camera.fovDegrees), aspect, 0.1f, 120.0f
	);

	// В Vulkan ось Y в NDC направлена вниз, поэтому изображение отражается.
	// Отражение выполняется в проекции, а не в матрице вида: у проекции в
	// строке Y всего одно ненулевое значение, поэтому знак меняется у всей
	// строки разом. Отражение одного элемента матрицы вида (как это было в
	// лабораторной №1) при наклонной камере искажает базис вида и ломает
	// winding у части граней: они отсекаются поодиночке.
	projection[1][1] *= -1.0f;

	const glm::mat4 viewProjection = projection * view;

	// Трансформация главного объекта нужна и для «прикреплённых» источников.
	const glm::mat4 pyramidModel = pyramidModelMatrix();

	// ---- Сборка данных источников света ----
	std::vector<GpuPointLight> gpuPointLights;
	std::vector<GpuSpotLight> gpuSpotLights;

	gpuPointLights.reserve(pointLights.size());
	gpuSpotLights.reserve(spotLights.size());

	for (const PointLight& light : pointLights) {
		if (!light.enabled) {
			continue;
		}

		const glm::vec3 worldPosition = light.attachedToObject
			? glm::vec3(pyramidModel * glm::vec4(light.position, 1.0f))
			: light.position;

		gpuPointLights.push_back(GpuPointLight{
			.positionRange = glm::vec4(worldPosition, light.range),
			.colorIntensity = glm::vec4(light.color, light.intensity)
		});
	}

	for (const SpotLight& light : spotLights) {
		if (!light.enabled) {
			continue;
		}

		const glm::vec3 worldPosition = light.attachedToObject
			? glm::vec3(pyramidModel * glm::vec4(light.position, 1.0f))
			: light.position;

		// Направление луча: от источника к сцене.
		const glm::vec3 direction = directionFromAngles(light.yawDegrees, light.pitchDegrees);

		gpuSpotLights.push_back(GpuSpotLight{
			.positionRange = glm::vec4(worldPosition, light.range),
			.directionCosOuter = glm::vec4(
				direction, std::cos(glm::radians(light.outerAngleDegrees))
			),
			.colorIntensity = glm::vec4(light.color, light.intensity),
			.innerCos = glm::vec4(std::cos(glm::radians(light.innerAngleDegrees)), 0.0f, 0.0f,
			                      0.0f)
		});
	}

	// Storage-буферы источников света. Если источников стало больше, буфер
	// пересоздаётся, а дескриптор обновляется.
	if (!ensureStorageCapacity(pointLightBuffer, 1,
	                           gpuPointLights.size() * sizeof(GpuPointLight))) {
		return;
	}

	if (!ensureStorageCapacity(spotLightBuffer, 2,
	                           gpuSpotLights.size() * sizeof(GpuSpotLight))) {
		return;
	}

	if (!gpuPointLights.empty()) {
		std::memcpy(pointLightBuffer.mapped, gpuPointLights.data(),
		            gpuPointLights.size() * sizeof(GpuPointLight));
	}

	if (!gpuSpotLights.empty()) {
		std::memcpy(spotLightBuffer.mapped, gpuSpotLights.data(),
		            gpuSpotLights.size() * sizeof(GpuSpotLight));
	}

	// ---- Данные сцены ----
	const GpuSceneBlock sceneBlock{
		.viewProjection = viewProjection,
		.cameraPosition = glm::vec4(camera.position, 1.0f),
		.ambientColor = glm::vec4(state.ambientColor, state.ambientIntensity),
		// Направленный источник: направление задано к источнику.
		.directionalDirection = glm::vec4(
			directionFromAngles(state.directionalYawDegrees, state.directionalPitchDegrees),
			state.directionalIntensity
		),
		.directionalColor = glm::vec4(state.directionalColor, 1.0f),
		.lightCounts = glm::vec4(
			static_cast<float>(gpuPointLights.size()),
			static_cast<float>(gpuSpotLights.size()),
			0.0f, 0.0f
		)
	};

	std::memcpy(sceneUniformBuffer.mapped, &sceneBlock, sizeof(sceneBlock));

	// ---- Список объектов ----
	litItems.clear();
	unlitItems.clear();
	transparentItems.clear();

	litItems.push_back(DrawItem{
		.mesh = &pyramidMesh,
		.model = pyramidModel,
		.albedo = glm::vec4(state.pyramidMaterial.albedo, 1.0f),
		.specular = glm::vec4(state.pyramidMaterial.specularColor,
		                      state.pyramidMaterial.shininess)
	});

	litItems.push_back(DrawItem{
		.mesh = &floorMesh,
		.model = glm::mat4(1.0f),
		.albedo = glm::vec4(state.floorMaterial.albedo, 1.0f),
		.specular = glm::vec4(state.floorMaterial.specularColor,
		                      state.floorMaterial.shininess)
	});

	// Визуализация источников света: постоянный цвет, без освещения.
	if (state.showLightVisualization) {
		for (const PointLight& light : pointLights) {
			if (!light.enabled) {
				continue;
			}

			const glm::vec3 worldPosition = light.attachedToObject
				? glm::vec3(pyramidModel * glm::vec4(light.position, 1.0f))
				: light.position;

			const glm::mat4 model = glm::translate(glm::mat4(1.0f), worldPosition) *
			                        glm::scale(glm::mat4(1.0f), glm::vec3(0.22f));

			unlitItems.push_back(DrawItem{
				.mesh = &cubeMesh,
				.model = model,
				.albedo = glm::vec4(light.color, 1.0f)
			});
		}

		for (const SpotLight& light : spotLights) {
			if (!light.enabled) {
				continue;
			}

			const glm::vec3 worldPosition = light.attachedToObject
				? glm::vec3(pyramidModel * glm::vec4(light.position, 1.0f))
				: light.position;

			const glm::vec3 direction = directionFromAngles(light.yawDegrees, light.pitchDegrees);

			// Базис, в котором локальная +Z совпадает с направлением луча.
			glm::vec3 side = glm::cross(direction, glm::vec3(0.0f, 1.0f, 0.0f));
			if (glm::dot(side, side) < 1e-8f) {
				side = glm::vec3(1.0f, 0.0f, 0.0f);
			} else {
				side = glm::normalize(side);
			}

			const glm::vec3 sideUp = glm::cross(direction, side);

			const glm::mat4 orientation{
				glm::vec4(side, 0.0f),
				glm::vec4(sideUp, 0.0f),
				glm::vec4(direction, 0.0f),
				glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
			};

			// Конус показывает направление и угол луча. Он намеренно небольшой:
			// иначе перекрывает сцену.
			const float coneLength = std::clamp(light.range * 0.08f, 0.6f, 1.5f);
			const float coneRadius = coneLength *
			                         std::tan(glm::radians(light.outerAngleDegrees));

			const glm::mat4 model = glm::translate(glm::mat4(1.0f), worldPosition) *
			                        orientation *
			                        glm::scale(glm::mat4(1.0f),
			                                   glm::vec3(coneRadius, coneRadius, coneLength));

			transparentItems.push_back(DrawItem{
				.mesh = &coneMesh,
				.model = model,
				.albedo = glm::vec4(light.color, 0.12f)
			});

			const glm::mat4 lampModel = glm::translate(glm::mat4(1.0f), worldPosition) *
			                            glm::scale(glm::mat4(1.0f), glm::vec3(0.22f));

			unlitItems.push_back(DrawItem{
				.mesh = &cubeMesh,
				.model = lampModel,
				.albedo = glm::vec4(light.color, 1.0f)
			});
		}
	}

	// ---- Запись команд ----
	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	const VkClearValue clears[2] = {
		{ .color = { { 0.04f, 0.05f, 0.07f, 1.0f } } },
		{ .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};

	const VkRenderPassBeginInfo renderPassBegin{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clears
	};
	vkCmdBeginRenderPass(fd.command_buffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport{
		0.0f, 0.0f,
		float(context.swapchain_extent.width), float(context.swapchain_extent.height),
		0.0f, 1.0f
	};
	const VkRect2D scissor{ .offset = { 0, 0 }, .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	// Набор дескрипторов сцены общий для всех объектов.
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
	                         0, 1, &sceneDescriptorSet, 0, nullptr);

	const VkDeviceSize vertexOffset = 0;

	// Счётчик слотов общего буфера моделей: каждому объекту — свой слот.
	uint32_t slot = 0;

	auto drawItems = [&](const std::vector<DrawItem>& items, VkPipeline pipeline) {
		if (items.empty()) {
			return;
		}

		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		for (const DrawItem& item : items) {
			if (slot >= maxModelSlots) {
				break;
			}

			const GpuModelBlock modelBlock{
				.model = item.model,
				.normalMatrix = glm::mat4(
					glm::transpose(glm::inverse(glm::mat3(item.model)))
				),
				.albedo = item.albedo,
				.specular = item.specular
			};

			// Смещение в динамическом дескрипторе измеряется в байтах и должно
			// быть кратно минимальному выравниванию uniform-буфера.
			const uint32_t dynamicOffset = static_cast<uint32_t>(slot * modelSlotStride);

			std::memcpy(static_cast<char*>(modelUniformBuffer.mapped) +
			                static_cast<size_t>(dynamicOffset),
			            &modelBlock, sizeof(modelBlock));

			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			                         pipelineLayout, 1, 1, &modelDescriptorSet, 1,
			                         &dynamicOffset);

			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &item.mesh->buffer.buffer,
			                       &vertexOffset);
			vkCmdDraw(fd.command_buffer, item.mesh->vertexCount, 1, 0, 0);

			++slot;
		}
	};

	drawItems(litItems, scenePipeline);
	drawItems(unlitItems, unlitPipeline);
	drawItems(transparentItems, unlitBlendPipeline);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
