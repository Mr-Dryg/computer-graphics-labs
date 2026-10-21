#pragma once

#include "graphics_internal.hpp"

#include <glm/glm.hpp>

struct GLFWwindow;

namespace application {

// Точечный источник света: освещает сцену во все стороны, интенсивность
// падает по закону обратных квадратов.
struct PointLight {
	glm::vec3 position{0.0f, 2.0f, 0.0f};
	glm::vec3 color{1.0f, 0.9f, 0.78f};
	float intensity{8.0f};
	float range{8.0f};
	// Если включено, источник движется вместе с объектом: position тогда
	// задаётся в локальных координатах объекта.
	bool attachedToObject{false};
	bool enabled{true};
};

// Прожекторный источник света: светит только внутри конуса. Края конуса
// мягкие: между внешним и внутренним углом интенсивность плавно нарастает.
struct SpotLight {
	glm::vec3 position{0.0f, 5.0f, 0.0f};
	glm::vec3 color{0.85f, 0.9f, 1.0f};
	float intensity{25.0f};
	float range{16.0f};
	float yawDegrees{0.0f};
	float pitchDegrees{-65.0f};
	float outerAngleDegrees{38.0f};
	float innerAngleDegrees{20.0f};
	bool attachedToObject{false};
	bool enabled{true};
};

// Параметры материала по модели Блинна-Фонга.
struct Material {
	glm::vec3 albedo{0.85f, 0.62f, 0.32f};
	glm::vec3 specularColor{0.55f, 0.55f, 0.55f};
	// Степень блеска: чем больше значение, тем меньше площадь блика.
	float shininess{32.0f};
};

// Параметры камеры.
struct Camera {
	glm::vec3 position{0.0f, 2.2f, 9.0f};
	float yawDegrees{0.0f};
	float pitchDegrees{-12.0f};
	float fovDegrees{45.0f};
};

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

void attachWindow(GLFWwindow* window);
void onMouseButton(int button, int action);
void onCursorPos(double xpos, double ypos);
void onScroll(double yoffset);

} // namespace application