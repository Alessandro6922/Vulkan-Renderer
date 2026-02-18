#version 450

layout(binding = 3) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColour;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 viewDirection;

layout(location = 0) out vec4 outColour;

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

const vec3 lightDirectionDark = vec3(-0.1, -0.1, 1.0);
const vec3 lightDirection = vec3(0.0, -1.0, 0.0);
const vec3 sunColour = vec3(1.0, 1.0, 1.0);
const vec3 ambientColourDark = vec3(0.12, 0.11, 0.16);
const vec3 ambientColour = vec3(0.44, 0.64, 0.92);
const float ambientLight = 0.9;
const float specularStrength = 0.7;

void main() {
	vec3 finalColour = vec3(1.0);

	if(int(gdbo.camPosition.w) == 0){
		vec3 surfaceNormal = normalize(normal);
	//	if(surfaceNormal.z < 0){
	//		surfaceNormal = -surfaceNormal;
	//	}
		vec3 lightDir = -lightDirection;

		vec4 texColour = texture(texSampler, fragTexCoord);
		texColour.rgb *= ambientColour;
		float intesity = normalize(dot(normal, lightDir));
		float backsideIntensity = normalize(dot(-normal, lightDir));
		//vec3 finalColour = normalize(sunColour * max(intesity, backsideIntensity));
		vec3 reflectDir = reflect(lightDir, surfaceNormal);
		float spec = pow(max(dot(viewDirection, reflectDir), 0.0), 128);
		vec3 specular = specularStrength * spec * sunColour;
		vec3 ambient = ambientLight * ambientColour;
		vec3 diffuse = normalize(sunColour * max(intesity, backsideIntensity));

		finalColour = (ambient + diffuse + specular) * texColour.rgb;
	}
	else if(int(gdbo.camPosition.w) == 1){
		vec4 texColour = texture(texSampler, fragTexCoord);
		finalColour = texColour.rgb;
	}
	else {
		finalColour = fragColour;
	}
	//outColour = vec4(texColour.xyz * vec3(0.25, 0.2, 0.35), 1.0f) * 0.3;
	outColour = vec4(finalColour, 1.0f);
	//outColour = vec4(clumpNormal, 1.0f);
}