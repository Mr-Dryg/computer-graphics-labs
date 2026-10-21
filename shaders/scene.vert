#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

// Набор дескрипторов 0 — общие данные сцены.
layout(set = 0, binding = 0) uniform SceneBlock {
	mat4 viewProjection;
	vec4 cameraPosition;
	vec4 ambientColor;
	vec4 directionalDirection;
	vec4 directionalColor;
	// x — число точечных источников, y — число прожекторных.
	vec4 lightCounts;
} sceneBlock;

// Набор дескрипторов 1 — данные конкретной модели.
layout(set = 1, binding = 0) uniform ModelBlock {
	mat4 model;
	mat4 normalMatrix;
	vec4 albedo;
	vec4 specular;
} modelBlock;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

void main() {
	vec4 worldPos = modelBlock.model * vec4(inPosition, 1.0);

	worldPosition = worldPos.xyz;
	worldNormal = normalize((modelBlock.normalMatrix * vec4(inNormal, 0.0)).xyz);

	gl_Position = sceneBlock.viewProjection * worldPos;
}
