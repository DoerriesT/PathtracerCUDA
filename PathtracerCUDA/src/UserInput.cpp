#include "UserInput.h"
#include "ContainerUtility.h"

UserInput::UserInput()
{
}

void UserInput::input()
{
	m_mousePosDelta[0] = (m_mousePos[0] - m_previousMousePos[0]);
	m_mousePosDelta[1] = (m_mousePos[1] - m_previousMousePos[1]);
	m_previousMousePos[0] = m_mousePos[0];
	m_previousMousePos[1] = m_mousePos[1];
}

void UserInput::getPreviousMousePos(float &x, float &y) const
{
	x = m_previousMousePos[0];
	y = m_previousMousePos[1];
}

void UserInput::getCurrentMousePos(float &x, float &y) const
{
	x = m_mousePos[0];
	y = m_mousePos[1];
}

void UserInput::getMousePosDelta(float &x, float &y) const
{
	x = m_mousePosDelta[0];
	y = m_mousePosDelta[1];
}

void UserInput::getScrollOffset(float &x, float &y) const
{
	x = m_scrollOffse[0];
	y = m_scrollOffse[1];
}

bool UserInput::isKeyPressed(InputKey key, bool ignoreRepeated) const
{
	size_t pos = static_cast<size_t>(key);
	return pos < m_repeatedKeys.size() && pos < m_repeatedKeys.size() && m_pressedKeys[pos] && (!ignoreRepeated || !m_repeatedKeys[pos]);
}

bool UserInput::isMouseButtonPressed(InputMouse mouseButton) const
{
	return m_pressedMouseButtons[static_cast<size_t>(mouseButton)];
}

void UserInput::addKeyListener(IKeyListener *listener)
{
	m_keyListeners.push_back(listener);
}

void UserInput::removeKeyListener(IKeyListener *listener)
{
	ContainerUtility::remove(m_keyListeners, listener);
}

void UserInput::addCharListener(ICharListener *listener)
{
	m_charListeners.push_back(listener);
}

void UserInput::removeCharListener(ICharListener *listener)
{
	ContainerUtility::remove(m_charListeners, listener);
}

void UserInput::addScrollListener(IScrollListener *listener)
{
	m_scrollListeners.push_back(listener);
}

void UserInput::removeScrollListener(IScrollListener *listener)
{
	ContainerUtility::remove(m_scrollListeners, listener);
}

void UserInput::addMouseButtonListener(IMouseButtonListener *listener)
{
	m_mouseButtonlisteners.push_back(listener);
}

void UserInput::removeMouseButtonListener(IMouseButtonListener *listener)
{
	ContainerUtility::remove(m_mouseButtonlisteners, listener);
}

void UserInput::onKey(InputKey key, InputAction action)
{
	for (IKeyListener *listener : m_keyListeners)
	{
		listener->onKey(key, action);
	}

	const auto keyIndex = static_cast<size_t>(key);

	switch (action)
	{
	case InputAction::RELEASE:
		if (keyIndex < m_pressedKeys.size() && keyIndex < m_repeatedKeys.size())
		{
			m_pressedKeys.set(static_cast<size_t>(key), false);
			m_repeatedKeys.set(static_cast<size_t>(key), false);
		}
		break;
	case InputAction::PRESS:
		if (keyIndex < m_pressedKeys.size())
		{
			m_pressedKeys.set(static_cast<size_t>(key), true);
		}
		break;
	case InputAction::REPEAT:
		if (keyIndex < m_repeatedKeys.size())
		{
			m_repeatedKeys.set(static_cast<size_t>(key), true);
		}
		break;
	default:
		break;
	}
}

void UserInput::onChar(Codepoint charKey)
{
	for (ICharListener *listener : m_charListeners)
	{
		listener->onChar(charKey);
	}
}

void UserInput::onMouseButton(InputMouse mouseButton, InputAction action)
{
	for (IMouseButtonListener *listener : m_mouseButtonlisteners)
	{
		listener->onMouseButton(mouseButton, action);
	}

	if (action == InputAction::RELEASE)
	{
		m_pressedMouseButtons.set(static_cast<size_t>(mouseButton), false);
	}
	else if (action == InputAction::PRESS)
	{
		m_pressedMouseButtons.set(static_cast<size_t>(mouseButton), true);
	}
}

void UserInput::onMouseMove(double x, double y)
{
	m_mousePos[0] = static_cast<float>(x);
	m_mousePos[1] = static_cast<float>(y);
}

void UserInput::onMouseScroll(double xOffset, double yOffset)
{
	for (IScrollListener *listener : m_scrollListeners)
	{
		listener->onMouseScroll(xOffset, yOffset);
	}

	m_scrollOffse[0] = static_cast<float>(xOffset);
	m_scrollOffse[1] = static_cast<float>(yOffset);
}
