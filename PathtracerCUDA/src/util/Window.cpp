#include "Window.h"
#include <GLFW/glfw3.h>

void windowSizeCallback(GLFWwindow *window, int width, int height);
void curserPosCallback(GLFWwindow *window, double xPos, double yPos);
void curserEnterCallback(GLFWwindow *window, int entered);
void scrollCallback(GLFWwindow *window, double xOffset, double yOffset);
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void charCallback(GLFWwindow *window, unsigned int codepoint);
void joystickCallback(int joystickId, int event);

Window::Window(unsigned int width, unsigned int height, const std::string &title)
	:m_windowHandle(),
	m_width(width),
	m_height(height),
	m_title(title),
	m_configurationChanged()
{
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 0);
	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);

	m_windowHandle = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);

	int w, h;
	glfwGetFramebufferSize(m_windowHandle, &w, &h);
	m_width = w;
	m_height = h;

	if (!m_windowHandle)
	{
		glfwTerminate();
		printf("Failed to create GLFW window\n");
		exit(EXIT_FAILURE);
		return;
	}

	glfwMakeContextCurrent(m_windowHandle);

	glfwSetFramebufferSizeCallback(m_windowHandle, windowSizeCallback);
	glfwSetCursorPosCallback(m_windowHandle, curserPosCallback);
	glfwSetCursorEnterCallback(m_windowHandle, curserEnterCallback);
	glfwSetScrollCallback(m_windowHandle, scrollCallback);
	glfwSetMouseButtonCallback(m_windowHandle, mouseButtonCallback);
	glfwSetKeyCallback(m_windowHandle, keyCallback);
	glfwSetCharCallback(m_windowHandle, charCallback);
	glfwSetJoystickCallback(joystickCallback);

	glfwSetWindowUserPointer(m_windowHandle, this);
}

Window::~Window()
{
	glfwDestroyWindow(m_windowHandle);
	glfwTerminate();
}

void Window::pollEvents()
{
	m_configurationChanged = false;
	glfwPollEvents();
}

void Window::present()
{
	glfwSwapBuffers(m_windowHandle);
}

void *Window::getWindowHandle() const
{
	return static_cast<void *>(m_windowHandle);
}

unsigned int Window::getWidth() const
{
	return m_width;
}

unsigned int Window::getHeight() const
{
	return m_height;
}

bool Window::shouldClose() const
{
	return glfwWindowShouldClose(m_windowHandle);
}

bool Window::configurationChanged() const
{
	return m_configurationChanged;
}

void Window::grabMouse(bool grabMouse)
{
	if (grabMouse)
	{
		glfwSetInputMode(m_windowHandle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else
	{
		glfwSetInputMode(m_windowHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void Window::setTitle(const std::string &title)
{
	m_title = title;
	glfwSetWindowTitle(m_windowHandle, m_title.c_str());
}

void Window::addInputListener(IInputListener *listener)
{
	m_inputListeners.push_back(listener);
}

void Window::removeInputListener(IInputListener *listener)
{
	auto it = std::find(m_inputListeners.begin(), m_inputListeners.end(), listener);
	if (it != m_inputListeners.end())
	{
		std::swap(m_inputListeners.back(), *it);
		m_inputListeners.erase(--m_inputListeners.end());
	}
}

// callback functions

void windowSizeCallback(GLFWwindow *window, int width, int height)
{
	Window *windowFramework = static_cast<Window *>(glfwGetWindowUserPointer(window));
	windowFramework->m_width = width;
	windowFramework->m_height = height;
	windowFramework->m_configurationChanged = true;
}

void curserPosCallback(GLFWwindow *window, double xPos, double yPos)
{
	Window *windowFramework = static_cast<Window *>(glfwGetWindowUserPointer(window));
	for (IInputListener *listener : windowFramework->m_inputListeners)
	{
		listener->onMouseMove(xPos, yPos);
	}
}

void curserEnterCallback(GLFWwindow *window, int entered)
{
}

void scrollCallback(GLFWwindow *window, double xOffset, double yOffset)
{
	Window *windowFramework = static_cast<Window *>(glfwGetWindowUserPointer(window));
	for (IInputListener *listener : windowFramework->m_inputListeners)
	{
		listener->onMouseScroll(xOffset, yOffset);
	}
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
	Window *windowFramework = static_cast<Window *>(glfwGetWindowUserPointer(window));
	for (IInputListener *listener : windowFramework->m_inputListeners)
	{
		listener->onMouseButton(static_cast<InputMouse>(button), static_cast<InputAction>(action));
	}
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	Window *windowFramework = static_cast<Window *>(glfwGetWindowUserPointer(window));
	for (IInputListener *listener : windowFramework->m_inputListeners)
	{
		listener->onKey(static_cast<InputKey>(key), static_cast<InputAction>(action));
	}
}

void charCallback(GLFWwindow *window, unsigned int codepoint)
{
	Window *windowFramework = static_cast<Window *>(glfwGetWindowUserPointer(window));
	for (IInputListener *listener : windowFramework->m_inputListeners)
	{
		listener->onChar(codepoint);
	}
}

void joystickCallback(int joystickId, int event)
{
	if (event == GLFW_CONNECTED)
	{

	}
	else if (event == GLFW_DISCONNECTED)
	{

	}
}
