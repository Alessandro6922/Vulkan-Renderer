#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(binding = 1) uniform GrassDataBufferObject{
	float instancesPerAxis;
	float spacing;
	float grassHeight;
	float bladeThickness;
	vec4 bezierCPoint1;
	vec4 bezierCPoint2;
	vec4 bezierEndPoint;
} gdbo;

layout(std430, binding = 2) readonly buffer GrassPositionsSSBOIn{
	vec4 position[];
} ssbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColour;
layout(location = 1) out vec2 fragTexCoord;

// https://gist.github.com/yiwenl/bb87c2d7df8bc735dab017c808a381ab
// two point bezier curve function
// A is startPoint, B & C are control points, D is the endPoint, T is from 0 to 1
vec3 bezier(vec3 A, vec3 B, vec3 C, vec3 D, float t) {
  vec3 E = mix(A, B, t);
  vec3 F = mix(B, C, t);
  vec3 G = mix(C, D, t);

  vec3 H = mix(E, F, t);
  vec3 I = mix(F, G, t);

  vec3 P = mix(H, I, t);

  return P;
}

vec4 rotateAroundYAxis(vec4 vertexPos, float deg){
	float alpha = deg * 3.1415 / 180;
	float sinA = sin(alpha);
	float cosA = cos(alpha);
	mat2x2 m = mat2x2(cosA, -sinA, sinA, cosA);
	return vec4((m * vertexPos.xz), vertexPos.yw).xzyw;
}

// i have no original source for this demon one liner its just a thing that gets done in glsl it would seem
// i found it here https://community.khronos.org/t/random-values/75728 and adapted for a instanceIndex so that all the blades vertices are rotated the same
float randomAngle(int instanceIndex){
	return fract(sin(float(instanceIndex) * 12.9898) * 43758.5453) * 360.0;
}

void main() {
vec3 instancePos = inPosition;
	instancePos.z = bezier(vec3(0,0,0), gdbo.bezierCPoint1.xyz, gdbo.bezierCPoint2.xyz, gdbo.bezierEndPoint.xyz, instancePos.y / gdbo.bezierEndPoint.w).z;
	instancePos *= vec3(gdbo.bladeThickness, 1.0, 1.0);

	vec4 rotatedPos = rotateAroundYAxis(vec4(instancePos, 1.0f), randomAngle(int(ssbo.position[gl_InstanceIndex].w)));

	rotatedPos.x += ssbo.position[gl_InstanceIndex].x;
	rotatedPos.z += ssbo.position[gl_InstanceIndex].z;
	rotatedPos.y *= gdbo.grassHeight;
	rotatedPos.y += ssbo.position[gl_InstanceIndex].y;

	gl_Position = ubo.proj * ubo.view * ubo.model * rotatedPos;
	fragColour = inColour;
	fragTexCoord = inTexCoord;
}