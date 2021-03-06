#pragma once
#include <vector>
#include "IInputListener.h"
#include <string>

struct GLFWwindow;

class Window
{
	friend void windowSizeCallback(GLFWwindow *window, int width, int height);
	friend void curserPosCallback(GLFWwindow *window, double xPos, double yPos);
	friend void curserEnterCallback(GLFWwindow *window, int entered);
	friend void scrollCallback(GLFWwindow *window, double xOffset, double yOffset);
	friend void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
	friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
	friend void charCallback(GLFWwindow *window, unsigned int codepoint);
	friend void joystickCallback(int joystickId, int event);
public:
	explicit Window(unsigned int width, unsigned int height, const std::string &title);
	~Window();
	void pollEvents();
	void present();
	void *getWindowHandle() const;
	unsigned int getWidth() const;
	unsigned int getHeight() const;
	bool shouldClose() const;
	bool configurationChanged() const;
	void grabMouse(bool grabMouse);
	void setTitle(const std::string &title);
	void addInputListener(IInputListener *listener);
	void removeInputListener(IInputListener *listener);

private:
	GLFWwindow *m_windowHandle;
	unsigned int m_width;
	unsigned int m_height;
	std::string m_title;
	std::vector<IInputListener *> m_inputListeners;
	bool m_configurationChanged;
};