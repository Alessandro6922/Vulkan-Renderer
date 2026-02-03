#version 450

layout(binding = 3) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColour;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColour;

void main() {
	vec4 texColour = texture(texSampler, fragTexCoord * 15);
	outColour = vec4(texColour.xyz * vec3(0.25, 0.2, 0.35), 1.0f) * 0.3;
}