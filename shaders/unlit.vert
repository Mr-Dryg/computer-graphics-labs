#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform SceneBlock {
	mat4 viewProjection;
	vec4 cameraPosition;
	vec4 ambientColor;
	vec4 directionalDirection;
	vec4 directionalColor;
	vec4 lightCounts;
} sceneBlock;

layout(set = 1, binding = 0) uniform ModelBlock {
	mat4 model;
	mat4 normalMatrix;
	vec4 albedo;
	vec4 specular;
} modelBlock;

layout(location = 0) out vec4 vertexColor;

void main() {
	// Фигуры визуализации источников света имеют постоянный цвет
	// и не освещаются по модели Блинн-Фонга.
	vertexColor = modelBlock.albedo;

	gl_Position = sceneBlock.viewProjection * modelBlock.model * vec4(inPosition, 1.0);
}
