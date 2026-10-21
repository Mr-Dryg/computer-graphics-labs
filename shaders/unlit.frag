#version 450

layout(location = 0) in vec4 vertexColor;

layout(location = 0) out vec4 outColor;

void main() {
	// Альфа берётся из альбедо: прозрачные фигуры визуализации
	// (конус прожектора) смешиваются с фоном.
	outColor = vertexColor;
}
