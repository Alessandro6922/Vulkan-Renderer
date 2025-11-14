#version 450

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(binding = 1) uniform sampler2D displacementSampler;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColour;
layout(location = 1) out vec2 fragTexCoord;

void main() {
	vec4 vertexPos = vec4(inPosition, 1.0f);
	vertexPos.y += 50 * texture(displacementSampler, inTexCoord).g;
	gl_Position = ubo.proj * ubo.view * ubo.model * vertexPos;
	fragColour = inColour;
	fragTexCoord = inTexCoord;
}