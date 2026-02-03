#version 450

layout(binding = 3) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColour;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 clumpNormal;

layout(location = 0) out vec4 outColour;

const vec3 lightDirection = vec3(1.0, -0.5, 0.2);

void main() {
	vec3 surfaceNormal = normalize(normal);
//	if(surfaceNormal.z < 0){
//		surfaceNormal = -surfaceNormal;
//	}
	vec4 texColour = texture(texSampler, fragTexCoord);
	//outColour = vec4(texColour.xyz * vec3(0.25, 0.2, 0.35), 1.0f) * 0.3;
	outColour = vec4(texColour.xyz, 1.0f);
	//outColour = vec4(clumpNormal, 1.0f);
}