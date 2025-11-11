#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	float instancesPerAxis;
	float spacing;
	float grassHeight;
	float bladeThickness;
	vec4 bezierCPoint1;
	vec4 bezierCPoint2;
	vec4 bezierEndPoint;
} ubo;

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

/*
	NOTES TO SELF:
		IT PROBABLY STARTS AT 0,0
		YOU WANT THE BEZIER CURVE TO GO IN THE X DIRECTION SO DEST SHOULD BE LIKE 1,1,0 OR SMTH
*/

void main() {
vec3 instancePos = inPosition;
	//instancePos.x = bezier(vec3(0,0,0), ubo.bezierCPoint1.xyz, ubo.bezierCPoint2.xyz, ubo.bezierEndPoint.xyz, instancePos.y / ubo.bezierEndPoint.w).x;
	instancePos *= vec3(ubo.bladeThickness, 1.0, 1.0);
	instancePos.x += ubo.spacing * (gl_InstanceIndex / ubo.instancesPerAxis);
	instancePos.z += ubo.spacing * (gl_InstanceIndex % int(ubo.instancesPerAxis));
	instancePos.y *= ubo.grassHeight;
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(instancePos, 1.0);
	fragColour = inColour;
	fragTexCoord = inTexCoord;
}