//////////////////
// AT SOMEPOINT WHEN YOU OBJECT ORIENT EVERYTHING THIS WILL BE WHERE IMGUI GETS HANDLED
//////////////////

#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <vector>
#include <iostream>

#include "External/ImGui/imgui.h"
#include "External/ImGui/imgui_impl_glfw.h"
#include "External/ImGui/imgui_impl_vulkan.h"

struct ImGUIInitInfo {
	VkInstance* instance;
	GLFWwindow* window;
	VkPhysicalDevice* physicalDevice;
	VkDevice* device;
	VkQueue* queue;
	VkDescriptorPool* descriptorPool;
	VkExtent2D swapchainExtent;
	VkFormat colorFormat;
	uint32_t imageCount;
};

class ImGUIRenderer
{
public:
	ImGUIRenderer();
	~ImGUIRenderer();

	void init(ImGUIInitInfo initInfo);
	void updateFrameSize(VkExtent2D swapchainExtent);

	VkCommandBuffer prepareCommandBuffer(int image);

private:
	void createDescriptorPool();

	void initImGUI();

	int frameWidth;
	int frameHeight;

	ImGUIInitInfo info;

	std::vector<VkCommandBuffer> commandBuffers;
	VkDescriptorPool descriptorPool;
};

