#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <Windows.h>

#define piDiv180 (M_PI / 180.0f)

#define Clamp(x, a, b) min(max((x),(a)), (b))
#define FloatLerp(x, y, s) (x)*(1.0f - (s)) + (y)*(s)
#define Float2Dot(ax, ay, bx, by) ((ax)*(bx) + (ay)*(by))
#define DegToRad(x) (x * piDiv180)