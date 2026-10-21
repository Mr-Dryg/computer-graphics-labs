#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;

// Точечный источник: xyz — позиция, w — радиус действия;
// rgb — цвет, w — интенсивность.
struct PointLight {
	vec4 positionRange;
	vec4 colorIntensity;
};

// Прожекторный источник: дополнительно хранит направление,
// косинусы внешнего и внутреннего углов для мягкого края конуса.
struct SpotLight {
	vec4 positionRange;
	vec4 directionCosOuter;
	vec4 colorIntensity;
	vec4 innerCos;
};

layout(set = 0, binding = 0) uniform SceneBlock {
	mat4 viewProjection;
	vec4 cameraPosition;
	vec4 ambientColor;
	vec4 directionalDirection;
	vec4 directionalColor;
	// x — число точечных источников, y — число прожекторных.
	vec4 lightCounts;
} sceneBlock;

// Storage-буферы: число элементов задаётся данными, а не константой
// в шейдере, поэтому массив можно менять без перекомпиляции.
layout(set = 0, binding = 1) readonly buffer PointLightBlock {
	PointLight lights[];
} pointLights;

layout(set = 0, binding = 2) readonly buffer SpotLightBlock {
	SpotLight lights[];
} spotLights;

layout(set = 1, binding = 0) uniform ModelBlock {
	mat4 model;
	mat4 normalMatrix;
	vec4 albedo;
	vec4 specular;
} modelBlock;

layout(location = 0) out vec4 outColor;

// Затухание по закону обратных квадратов с мягким срезом у границы радиуса.
float attenuation(float distanceToLight, float range) {
	const float distanceSquared = max(distanceToLight * distanceToLight, 1e-4);
	const float ratio = clamp(distanceToLight / max(range, 1e-3), 0.0, 1.0);
	const float cutoff = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);

	return cutoff / distanceSquared;
}

// Модель Блинна-Фонга: рассеянная часть + блик.
vec3 blinnPhong(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightColor,
                vec3 albedo, vec3 specularColor, float shininess) {
	const float nDotL = max(dot(normal, lightDir), 0.0);

	if (nDotL <= 0.0) {
		return vec3(0.0);
	}

	const vec3 halfVector = normalize(lightDir + viewDir);
	const float nDotH = max(dot(normal, halfVector), 0.0);
	const float specular = pow(nDotH, max(shininess, 1.0)) * nDotL;

	return lightColor * (albedo * nDotL + specularColor * specular);
}

void main() {
	const vec3 normal = normalize(worldNormal);
	const vec3 viewDir = normalize(sceneBlock.cameraPosition.xyz - worldPosition);
	const vec3 albedo = modelBlock.albedo.rgb;
	const vec3 specularColor = modelBlock.specular.rgb;
	const float shininess = modelBlock.specular.a;

	// Рассеянный (ambient) свет.
	vec3 color = sceneBlock.ambientColor.rgb * sceneBlock.ambientColor.a * albedo;

	// Направленный источник: затухания нет, направление задано сценой.
	{
		const vec3 lightDir = normalize(-sceneBlock.directionalDirection.xyz);
		const vec3 lightColor = sceneBlock.directionalColor.rgb * sceneBlock.directionalDirection.w;

		color += blinnPhong(normal, viewDir, lightDir, lightColor, albedo,
		                    specularColor * sceneBlock.directionalColor.a, shininess);
	}

	// Точечные источники: затухание по закону обратных квадратов.
	// Число элементов приходит из данных сцены: максимальный размер массива
	// в шейдере не задан, буфер может содержать любое количество источников.
	const int pointLightCount = int(sceneBlock.lightCounts.x);

	for (int i = 0; i < pointLightCount; ++i) {
		const PointLight light = pointLights.lights[i];
		const vec3 toLight = light.positionRange.xyz - worldPosition;
		const float distanceToLight = length(toLight);

		if (distanceToLight > light.positionRange.w) {
			continue;
		}

		const vec3 lightDir = toLight / max(distanceToLight, 1e-4);
		const vec3 lightColor = light.colorIntensity.rgb * light.colorIntensity.w
		                     * attenuation(distanceToLight, light.positionRange.w);

		color += blinnPhong(normal, viewDir, lightDir, lightColor, albedo,
		                    specularColor, shininess);
	}

	// Прожекторные источники: конус с мягкими краями и затуханием.
	const int spotLightCount = int(sceneBlock.lightCounts.y);

	for (int i = 0; i < spotLightCount; ++i) {
		const SpotLight light = spotLights.lights[i];
		const vec3 toLight = light.positionRange.xyz - worldPosition;
		const float distanceToLight = length(toLight);

		if (distanceToLight > light.positionRange.w) {
			continue;
		}

		const vec3 lightDir = toLight / max(distanceToLight, 1e-4);
		const vec3 direction = normalize(light.directionCosOuter.xyz);
		const float cosAngle = dot(-lightDir, direction);

		// Мягкий край: чем меньше cosOuter относительно cosInner,
		// тем шире переход между освещённой и неосвещённой зоной.
		const float cosInner = max(light.innerCos.x, light.directionCosOuter.w + 1e-4);
		const float spot = smoothstep(light.directionCosOuter.w, cosInner, cosAngle);

		if (spot <= 0.0) {
			continue;
		}

		const vec3 lightColor = light.colorIntensity.rgb * light.colorIntensity.w
		                     * attenuation(distanceToLight, light.positionRange.w) * spot;

		color += blinnPhong(normal, viewDir, lightDir, lightColor, albedo,
		                    specularColor, shininess);
	}

	outColor = vec4(color, 1.0);
}
