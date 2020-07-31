#pragma once

// holds general application parameters that can be configured from the command line
struct Params
{
	unsigned int m_width = 1024;
	unsigned int m_height = 1024;
	unsigned int m_spp = 1024;
	const char *m_inputFilepath = nullptr;
	const char *m_outputFilepath = nullptr;
	bool m_showWindow = false;
	bool m_enableControls = false;
	bool m_outputHdr = false;
};