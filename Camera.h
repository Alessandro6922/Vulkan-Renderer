#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include "External/glm/glm.hpp"
#include "External/glm/gtc/matrix_transform.hpp"
#include "External/glm/gtx/quaternion.hpp"

class Camera
{
public:
	Camera();
	~Camera();

	void update();
	void updateRotation(float changeX, float changeY);

	void setVelocity(glm::vec3 newVelocity);

	glm::vec3 getVelocity();
	glm::vec3 getPosition();
	float getYaw();
	float getPitch();
	float getSensitivity();
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();

	glm::mat4 getRotationMatrix();

private:
	glm::vec3 position;
	glm::vec3 velocity;

	float yaw, pitch;
	float sensitivity;

	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;
};

