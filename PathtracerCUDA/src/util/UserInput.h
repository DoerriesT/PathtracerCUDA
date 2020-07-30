#pragma once
#include "IInputListener.h"
#include <vector>
#include <bitset>

class UserInput :public IInputListener
{
public:
	explicit UserInput();
	UserInput(const UserInput &) = delete;
	UserInput(const UserInput &&) = delete;
	UserInput &operator= (const UserInput &) = delete;
	UserInput &operator= (const UserInput &&) = delete;
	void input();
	void getPreviousMousePos(float &x, float &y) const;
	void getCurrentMousePos(float &x, float &y) const;
	void getMousePosDelta(float &x, float &y) const;
	void getScrollOffset(float &x, float &y) const;
	bool isKeyPressed(InputKey key, bool ignoreRepeated = false) const;
	bool isMouseButtonPressed(InputMouse mouseButton) const;
	void addKeyListener(IKeyListener *listener);
	void removeKeyListener(IKeyListener *listener);
	void addCharListener(ICharListener *listener);
	void removeCharListener(ICharListener *listener);
	void addScrollListener(IScrollListener *listener);
	void removeScrollListener(IScrollListener *listener);
	void addMouseButtonListener(IMouseButtonListener *listener);
	void removeMouseButtonListener(IMouseButtonListener *listener);
	void onKey(InputKey key, InputAction action) override;
	void onChar(Codepoint charKey) override;
	void onMouseButton(InputMouse mouseButton, InputAction action) override;
	void onMouseMove(double x, double y) override;
	void onMouseScroll(double xOffset, double yOffset) override;

private:
	float m_mousePos[2];
	float m_previousMousePos[2];
	float m_mousePosDelta[2];
	float m_scrollOffse[2];
	std::vector<IKeyListener *> m_keyListeners;
	std::vector<ICharListener *> m_charListeners;
	std::vector<IScrollListener *> m_scrollListeners;
	std::vector<IMouseButtonListener *> m_mouseButtonlisteners;
	std::bitset<350> m_pressedKeys;
	std::bitset<350> m_repeatedKeys;
	std::bitset<8> m_pressedMouseButtons;
};