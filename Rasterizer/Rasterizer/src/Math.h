#pragma once

#include <cmath>
#include <Windows.h>

#define Clamp(x, a, b) min(max((x),(a)), (b))
#define FloatLerp(x, y, s) (x)*(1.0f - (s)) + (y)*(s)
#define Float2Dot(ax, ay, bx, by) ((ax)*(bx) + (ay)*(by))