#include "ImGUIRenderer.h"

ImGUIRenderer::ImGUIRenderer()
{
	frameWidth = 0.0f;
	frameHeight = 0.0f;

	descriptorPool = nullptr;
}

ImGUIRenderer::~ImGUIRenderer()
{
}

void ImGUIRenderer::init(ImGUIInitInfo initInfo)
{
	info = initInfo;

	updateFrameSize(info.swapchainExtent);

	createDescriptorPool();
	initImGUI();
}

void ImGUIRenderer::updateFrameSize(VkExtent2D swapchainExtent)
{
	frameHeight = swapchainExtent.height;
	frameWidth = swapchainExtent.width;
}

VkCommandBuffer ImGUIRenderer::prepareCommandBuffer(int image)
{
	return VkCommandBuffer();
}

void ImGUIRenderer::createDescriptorPool()
{
	VkDescriptorPoolSize poolSizes[] = {
		{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
		{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
		{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
		{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
	};

	VkDescriptorPoolCreateInfo poolCreateInfo{};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	poolCreateInfo.maxSets = 1000 * IM_ARRAYSIZE(poolSizes);
	poolCreateInfo.poolSizeCount = (uint32_t)IM_ARRAYSIZE(poolSizes);
	poolCreateInfo.pPoolSizes = poolSizes;

	if (vkCreateDescriptorPool(*info.device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create ImGUI Descriptor Pool");
	}
}

void ImGUIRenderer::initImGUI()
{
	//ImGui::CreateContext();

	//ImGuiIO& io = ImGui::GetIO();
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;
	//io.DisplaySize.x = (float)frameWidth;
	//io.DisplaySize.y = (float)frameHeight;

	//ImGui::GetStyle().FontScaleMain = 1.5f;

	//ImGui::StyleColorsDark();

	//bool installGLFWCallbacks = true;
	//ImGui_ImplGlfw_InitForVulkan(info.window, installGLFWCallbacks);

	////VkPipelineCreateInfoKHR pipelineInfo{};
	////pipelineInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
	////pipelineInfo.pNext = nullptr;
	////pipelineInfo.viewMask = 0;
	////pipelineInfo.colorAttachmentCount = 1;
	////pipelineInfo.pColorAttachmentFormats = &info.colorFormat;

	//ImGui_ImplVulkan_InitInfo initInfo{};
	//initInfo.ApiVersion = VK_API_VERSION_1_0;
	//initInfo.Instance = *info.instance;
	//initInfo.PhysicalDevice = *info.physicalDevice;
	//initInfo.Device = *info.device;
	//initInfo.QueueFamily = 1;
	//initInfo.Queue = *info.queue;
	//initInfo.DescriptorPool = *info.descriptorPool;
	//initInfo.MinImageCount = info.imageCount - 1;
	//initInfo.ImageCount = info.imageCount;

	//ImGui_ImplVulkan_Init(&initInfo);

	//commandBuffers.resize(3);
}
