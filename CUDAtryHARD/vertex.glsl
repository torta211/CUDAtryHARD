#version 330

// VS locations
#define POSITION	0
// FS locations
#define FRAG_COLOR	0

layout(location = POSITION) in vec4 Pin;

out block
{
	vec4 Position;
	vec3 Color;
} VS_Out;

void main()
{
	gl_PointSize = 10.0f;
	gl_Position = vec4(Pin.x/(1.5f + Pin.z)/5.0f, Pin.y/(1.5f + Pin.z)/5.0f, Pin.z, 1.0f);
	VS_Out.Position = gl_Position;
	VS_Out.Color = vec3(0.3f + (1.0f + Pin.z) / 3.0f, 0.45f, 0.65f);
}