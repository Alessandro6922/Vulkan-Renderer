#version 450

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColour;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 position;

void main() {
	vec4 vertexPos = vec4(inPosition, 1.0f);
	position = vertexPos.xyz;
	vertexPos.xyz *= 10000;
	gl_Position = ubo.proj * ubo.view * ubo.model * vertexPos;
	fragColour = inColour;
	fragTexCoord = inTexCoord;
}