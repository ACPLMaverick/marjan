#ifndef _INPUT_H_
#define _INPUT_H_
#endif


class Input
{
private:
	bool m_keys[256];
public:
	const float movementDistance = 0.25f;

	Input();
	Input(const Input&);
	~Input();

	void Initialize();
	void KeyDown(unsigned int);
	void KeyUp(unsigned int);
	bool IsKeyDown(unsigned int);
};

