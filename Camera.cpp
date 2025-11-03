#include "Camera.h"

Camera::Camera()
{
	position = glm::vec3(2.0f, 2.0f, 2.0f);
	velocity = glm::vec3(0.0f);
	yaw = 0.0f;
	pitch = 0.0f;

	sensitivity = 0.0005f;

	viewMatrix = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	projectionMatrix = glm::mat4(1.0f);
}

Camera::~Camera()
{
}

void Camera::update()
{
	//float radYaw = glm::radians(yaw);
	//float radPitch = glm::radians(pitch);

	//glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), radYaw, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), radPitch, glm::vec3(0.0f, 1.0f, 0.0f));
	//glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f)) * rotation;

	glm::mat4 camRot = getRotationMatrix();
	position += glm::vec3(camRot * glm::vec4(velocity * 0.0005f, 0.0f));

	//viewMatrix = glm::inverse(transform);
}

void Camera::updateRotation(float changeX, float changeY)
{
	yaw -= changeX;
	pitch += changeY;

	//glm::mat4 camRot = getRotationMatrix();
	//position += glm::vec3(camRot * glm::vec4(velocity * 0.5f, 0.0f));
}

void Camera::setVelocity(glm::vec3 newVelocity)
{
	velocity = newVelocity;
}

glm::vec3 Camera::getVelocity()
{
	return velocity;
}

glm::vec3 Camera::getPosition()
{
	return position;
}

float Camera::getYaw()
{
	return yaw;
}

float Camera::getPitch()
{
	return pitch;
}

float Camera::getSensitivity()
{
	return sensitivity;
}

glm::mat4 Camera::getViewMatrix()
{
	glm::mat4 camTranslation = glm::translate(glm::mat4(1.0f), position);
	glm::mat4 camRotation = getRotationMatrix();

	return glm::inverse(camTranslation * camRotation);
}

glm::mat4 Camera::getProjectionMatrix()
{
	return projectionMatrix;
}

glm::mat4 Camera::getRotationMatrix()
{
	glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3(0.0f, -1.0f, 0.0f));

	return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}
