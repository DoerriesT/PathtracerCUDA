#include "UserInput.h"
#include "ContainerUtility.h"

UserInput::UserInput()
{
}

void UserInput::input()
{
	m_mousePosDelta = (m_mousePos - m_previousMousePos);
	m_previousMousePos = m_mousePos;
}

glm::vec2 UserInput::getPreviousMousePos() const
{
	return m_previousMousePos;
}

glm::vec2 UserInput::getCurrentMousePos() const
{
	return m_mousePos;
}

glm::vec2 UserInput::getMousePosDelta() const
{
	return m_mousePosDelta;
}

glm::vec2 UserInput::getScrollOffset() const
{
	return m_scrollOffset;
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
	m_mousePos.x = static_cast<float>(x);
	m_mousePos.y = static_cast<float>(y);
}

void UserInput::onMouseScroll(double xOffset, double yOffset)
{
	for (IScrollListener *listener : m_scrollListeners)
	{
		listener->onMouseScroll(xOffset, yOffset);
	}

	m_scrollOffset.x = static_cast<float>(xOffset);
	m_scrollOffset.y = static_cast<float>(yOffset);
}
