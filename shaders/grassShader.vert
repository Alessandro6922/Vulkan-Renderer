#version 460
#extension GL_KHR_vulkan_glsl : require

const uint grassCount = 65536 * 16;
const float RANDOM_HEIGHT_SCALE = 0.7;
const float PI = 3.1415;

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(set = 0, binding = 1) uniform GrassDataBufferObject{
	float elapsedTime;
	float grassLean;
	float grassHeight;
	float bladeThickness;
	float curveStrength;
	float windLeanStrength;
	float windSpeed;
	float windOffsetStrength;
	float windDirection;
	float maxDistance;
	float padding1;
	float padding2;
	vec4 camPosition;
	vec4 bezierEndPoint;
} gdbo;

struct blade{
    vec4 position;
    vec2 worldSpaceUV;
};

layout(std430, set = 0, binding = 2) readonly buffer GrassPositionsSSBOIn{
	blade bladeInfo[grassCount];
} ssbo;

layout(binding = 4) uniform sampler2D noiseSampler;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColour;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 normalOut;

vec3 bezier(vec3 A, vec3 B, vec3 C, float t) {
  vec3 E = mix(A, B, t);
  vec3 F = mix(B, C, t);
  vec3 G = mix(E, F, t);

  return G;
}

vec3 bezierDerivative(vec3 p0, vec3 p1, vec3 p2, float t) {
    return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}

// https://github.com/klejah/ResponsiveGrassDemo/blob/6ce514717467acc80fd965a6f7695d5151ba8c03/ResponsiveGrassDemo/shader/Grass/GrassUpdateForcesShader.cs#L67
void MakePersistentLength(in vec3 groundPos, inout vec3 v1, inout vec3 v2, in float height) {
    //Persistent length
    vec3 v01 = v1 - groundPos;
    vec3 v12 = v2 - v1;
    float lv01 = length(v01);
    float lv12 = length(v12);


    float L1 = lv01 + lv12;
    float L0 = length(v2 - groundPos);
    float L = (2.0 * L0 + L1) / 3.0; //http://steve.hollasch.net/cgindex/curves/cbezarclen.html

    float ldiff = height / L;
    v01 = v01 * ldiff;
    v12 = v12 * ldiff;
    v1 = groundPos + v01;
    v2 = v1 + v12;
}

//utils https://github.com/GPUOpen-LibrariesAndSDKs/WorkGraphsMeshNodeSample/blob/main/meshNodeSample/shaders/utils.hlsl

uint Hash(uint seed)
{
	seed = (seed ^ 61u) ^ (seed >> 16u);
	seed *= 9u;
	seed = seed ^ (seed >> 4u);
	seed *= 0x27d4eb2du;
	seed = seed ^ (seed >> 15u);
	return seed;
}

uint CombineSeed(uint a, uint b)
{
	return a ^ Hash(b) + 0x9e3779b9 + (a << 6) + (a >> 2);
}

uint CombineSeed(uint a, uint b, uint c)
{
	return CombineSeed(CombineSeed(a, b), c);
}

uint CombineSeed(uint a, uint b, uint c, uint d)
{
	return CombineSeed(CombineSeed(a, b), c, d);
}

uint Hash(in float seed)
{
	return Hash(uint(seed));
}

uint Hash(in vec3 vec)
{
	return CombineSeed(Hash(vec.x), Hash(vec.y), Hash(vec.z));
}

uint Hash(in vec4 vec)
{
	return CombineSeed(Hash(vec.x), Hash(vec.y), Hash(vec.z), Hash(vec.w));
}

float Random(uint seed)
{
	return Hash(seed) / float(~0u);
}

float Random(uint a, uint b)
{
	return Random(CombineSeed(a, b));
}

float Random(uint a, uint b, uint c)
{
	return Random(CombineSeed(a, b), c);
}

float Random(uint a, uint b, uint c, uint d)
{
	return Random(CombineSeed(a, b), c, d);
}

float Random(uint a, uint b, uint c, uint d, uint e)
{
	return Random(CombineSeed(a, b), c, d, e);
}

// https://gpuopen.com/learn/mesh_shaders/mesh_shaders-procedural_grass_rendering/
int tsign(in uint gtid, in int id) {
    return ((gtid & (1u << uint(id))) != 0) ? 1 : -1;
}

// https://gist.github.com/companje/29408948f1e8be54dd5733a74ca49bb9
float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

// https://www.shadertoy.com/view/wtcczf
float EaseInSine(float x)
{
    return 1.0 - cos((x * PI) / 2.0);
}

// Writing this comment like a month after adding this function but if i remember right i took
// it from a unity forum post talking about how its a built in unity function i believe
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
	vec3 vertPos = inPosition;
	const vec4 bladePos = ssbo.bladeInfo[gl_InstanceIndex].position;
	const vec2 bladeUV = ssbo.bladeInfo[gl_InstanceIndex].worldSpaceUV;
	const vec3 bladeNormal = vec3(0, 1, 0);

	const uint seed = CombineSeed(uint(bladePos.x), uint(bladePos.z));

	float grassLeanStrength = gdbo.grassLean * gdbo.windLeanStrength * texture(noiseSampler, (bladeUV * 3.0) - (gdbo.elapsedTime * vec2(cos(gdbo.windDirection), sin(gdbo.windDirection)) * gdbo.windSpeed)).r;
	float bladeDirectionAngle = mix(2.0 * PI * Random(seed, 4, gl_InstanceIndex), gdbo.windDirection, 0.9);
	vec2 bladeDirection = vec2(cos(bladeDirectionAngle), sin(bladeDirectionAngle)) * gdbo.grassHeight * grassLeanStrength;
	
	const float height = gdbo.grassHeight + float(Random(seed, gl_InstanceIndex, 20)) * RANDOM_HEIGHT_SCALE;

	vec3 tangent = normalize(cross(vec3(0, 0, 1), bladeNormal));
	vec3 biTangent = normalize(cross(bladeNormal, tangent));

	vec3 sideVec = normalize(vec3(bladeDirection.y, 0.0, -bladeDirection.x));
	vec3 offset = vertPos * gdbo.bladeThickness;

	vec3 p0 = bladePos.xyz;
	vec3 p2 = p0 + vec3(0.0, gdbo.grassHeight, 0.0) + vec3(bladeDirection.x, 0.0, bladeDirection.y);
	vec3 p1 = p0 + ((p2 - p0) * 0.75) + cross(((p2 - p0) * 0.75), sideVec) * gdbo.curveStrength;

	MakePersistentLength(p0, p1, p2, gdbo.grassHeight);

	//vec3 secondPoint = instancePos;
	//secondPoint.z = bezier(vec3(0,0,0), gdbo.bezierCPoint1.xyz, gdbo.bezierCPoint2.xyz, gdbo.bezierEndPoint.xyz, (instancePos.y / gdbo.bezierEndPoint.w) - 0.02).z;

	vec3 derivative = bezierDerivative(vec3(0,0,0), gdbo.bezierCPoint1.xyz, gdbo.bezierCPoint2.xyz, gdbo.bezierEndPoint.xyz, instancePos.y / gdbo.bezierEndPoint.w);
	vec3 normal = cross(normalize(derivative), vec3(1,0,0));
	//vec3 directionTwo = vec3(1,0,0);
	//vec3 normal = 
	instancePos *= vec3(gdbo.bladeThickness, 1.0, 1.0);
	instancePos += ((normalize(normal) * (sin((gdbo.elapsedTime + (instancePos.y / gdbo.bezierEndPoint.w) * 3.0 + gl_InstanceIndex) * 1.0)) - 0.5) * (instancePos.y / gdbo.bezierEndPoint.w) * 4 * texture(noiseSampler, vec2(ssbo.position[gl_InstanceIndex].x / 100.0 + gdbo.elapsedTime / 50.0, ssbo.position[gl_InstanceIndex].z / 100.0 + gdbo.elapsedTime / 50.0)).g);

	vec4 rotatedPos = rotateAroundYAxis(vec4(instancePos, 1.0f), randomAngle(int(ssbo.position[gl_InstanceIndex].w)));
	normalOut = normalize(rotateAroundYAxis(vec4(normal, 0.0f), randomAngle(int(ssbo.position[gl_InstanceIndex].w))).xyz);

	rotatedPos.x += ssbo.position[gl_InstanceIndex].x;
	rotatedPos.z += ssbo.position[gl_InstanceIndex].z;
	rotatedPos.y *= gdbo.grassHeight;
	rotatedPos.y += ssbo.position[gl_InstanceIndex].y;

	gl_Position = ubo.proj * ubo.view * ubo.model * rotatedPos;
	fragColour = inColour;
	fragTexCoord = inTexCoord;
}