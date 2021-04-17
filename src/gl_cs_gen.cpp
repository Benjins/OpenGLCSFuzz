
// Struct to represent a compute shader source

// Function that converts the AST-ish thing into GLSL

// Function that runs/interprets the AST-ish thing

// GL wrapper code that runs a single iteration, checks the results, returns info
// (save off diffs incl. the source code, GLSL binary, images, c code for CPU?, etc.)
// I guess will also need to supply random buffers...though idk


// Things compute shaders can do:
// - 4 arithmetic ops
// - some builtin functions
// - if statements
// - write to output
// - read from their pixel
// - read from other pixels (requires sync)
// - TODO: Shared? Atomics? How could we simulate those?


// Basic idea: you start with reading from the image input
// Then there's a "stream" thing where it is mostly reading
// from the last written value, doing some operation to it,
// and writing back to the same value...so weird SSA that
// doesn't need that much state...easy to emulate in CPU lol

#include <random>
#include <assert.h>


#include <GL/gl.h>

// NOTE: I know it's technically UB to read from a union member that
// overlaps with a different, previously written union member
// Look me in the eyes
// I do not care
struct Vec4f
{
	union
	{
		struct
		{
			float x;
			float y;
			float z;
			float w;
		};

		float xyzw[4];
	};

	float Magnitude() const
	{
		return sqrt(x * x + y * y + z * z + w * w);
	}

	Vec4f operator+(const Vec4f& other) const
	{
		Vec4f ret;
		for (int k = 0; k < 4; k++) { ret.xyzw[k] = xyzw[k] + other.xyzw[k]; }
		return ret;
	}

	Vec4f operator-(const Vec4f& other) const
	{
		Vec4f ret;
		for (int k = 0; k < 4; k++) { ret.xyzw[k] = xyzw[k] - other.xyzw[k]; }
		return ret;
	}

	Vec4f operator*(const Vec4f& other) const
	{
		Vec4f ret;
		for (int k = 0; k < 4; k++) { ret.xyzw[k] = xyzw[k] * other.xyzw[k]; }
		return ret;
	}
};

enum struct GLCSOperationType
{
	Add,
	Mul,
	Sub,
	Mov,
	WriteToOutput,
	BeginIf,
	EndIf
};

enum struct GLCSValueType
{
	Scratch,
	//Scalar,
	ReadFromInput,
	ReadFromOutput,
	ReadFromOutputFixed,
	VecLiteral
};


struct GLCSValue
{
	GLCSValueType Type = GLCSValueType::Scratch;

	Vec4f LiteralValue;
	int Index1 = 0;
	int Index2 = 0;
};

struct GLCSOperation
{
	GLCSOperationType Type = GLCSOperationType::Add;
	GLCSValue Value1;
	GLCSValue Value2;

	// Only for if statement
	struct
	{
		int ComponentIndex1 = 0;
		int ComponentIndex2 = 0;
	} IfStmt;
};

struct GLCSMetadata
{
	// TODO: Implement these
	int NumInputs = 1;
	int NumOutputs = 1;
};

void GenerateGLCSOperations(std::mt19937_64* RNGState, std::vector<GLCSOperation>* OutComputeOperations)
{

	auto GetGLCSValue = [RNGState](bool bCanReadFromOutput)
	{
		GLCSValue Value;

		uint64_t Decider = (*RNGState)() % 100;

		if (Decider < 20)
		{
			Value.Type = GLCSValueType::Scratch;
		}
		else if (Decider < 70)
		{
			Value.Type = GLCSValueType::ReadFromInput;
			Value.Index1 = (*RNGState)() % 4;
			Value.Index2 = (*RNGState)() % 4;
		}
		else if (bCanReadFromOutput && Decider < 80)
		{
			Value.Type = GLCSValueType::ReadFromOutputFixed;
		}
		else if (bCanReadFromOutput && Decider < 90)
		{
			// Somehow....it's this that's causing shifts...
			Value.Type = GLCSValueType::ReadFromOutput;
			Value.Index1 = (*RNGState)() % 4;
			Value.Index2 = (*RNGState)() % 4;
		}
		else if (Decider < 100)
		{
			Value.Type = GLCSValueType::VecLiteral;
			std::uniform_real_distribution<float> Dist(0.5f, 2.0f);
			for (int i = 0; i < 4; i++)
			{
				Value.LiteralValue.xyzw[i] = Dist(*RNGState);

				if ((*RNGState)() % 5 == 0)
				{
					Value.LiteralValue.xyzw[i] *= -1.0f;
				}
			}
		}
		else
		{
			assert(false && "Can't get here");
		}

		return Value;
	};

	int NumOps = std::uniform_int_distribution<int>(5, 25)(*RNGState);

	int IfStmtDepth = 0;

	bool bHasWrittenToOutput = false;

	for (int i = 0; i < NumOps; i++)
	{
		GLCSOperation Op;

		uint64_t Decider = (*RNGState)() % 100;

		if (Decider < 20)
		{
			Op.Type = GLCSOperationType::Add;
			Op.Value1.Type = GLCSValueType::Scratch;
			Op.Value2 = GetGLCSValue(bHasWrittenToOutput);
		}
		else if (Decider < 40)
		{
			Op.Type = GLCSOperationType::Mul;
			Op.Value1.Type = GLCSValueType::Scratch;
			Op.Value2 = GetGLCSValue(bHasWrittenToOutput);
		}
		else if (Decider < 60)
		{
			Op.Type = GLCSOperationType::Sub;
			Op.Value1.Type = GLCSValueType::Scratch;
			Op.Value2 = GetGLCSValue(bHasWrittenToOutput);
		}
		else if (IfStmtDepth < 4 && Decider < 80)
		{
			Op.Type = GLCSOperationType::BeginIf;
			Op.Value1.Type = GLCSValueType::Scratch;
			Op.Value2 = GetGLCSValue(bHasWrittenToOutput);
			Op.IfStmt.ComponentIndex1 = (*RNGState)() % 4;
			Op.IfStmt.ComponentIndex2 = (*RNGState)() % 4;
		
			IfStmtDepth++;
		}
		else if (IfStmtDepth > 0 && Decider < 90)
		{
			Op.Type = GLCSOperationType::EndIf;
			IfStmtDepth--;
		}
		else if (IfStmtDepth == 0 && Decider < 100)
		{
			Op.Type = GLCSOperationType::WriteToOutput;
			bHasWrittenToOutput = true;
		}
		else if (Decider < 100)
		{
			Op.Type = GLCSOperationType::Add;
			Op.Value1.Type = GLCSValueType::Scratch;
			Op.Value2 = GetGLCSValue(bHasWrittenToOutput);
		}
		else
		{
			assert(false && "Can't get here");
		}

		OutComputeOperations->push_back(Op);

		{
			GLCSOperation Op2;
			Op2.Type = GLCSOperationType::Mov;
			Op2.Value1.Type = GLCSValueType::ReadFromInput;
			Op2.Value1.Index1 = (*RNGState)() % 4;
			Op2.Value1.Index2 = (*RNGState)() % 4;
			OutComputeOperations->push_back(Op2);
		}
	}

	for (int i = 0; i < IfStmtDepth; i++)
	{
		GLCSOperation Op;
		Op.Type = GLCSOperationType::EndIf;
		OutComputeOperations->push_back(Op);
	}

	{
		GLCSOperation Op;
		Op.Type = GLCSOperationType::Mov;
		Op.Value1.Type = GLCSValueType::ReadFromInput;
		Op.Value1.Index1 = (*RNGState)() % 4;
		Op.Value1.Index2 = (*RNGState)() % 4;
		OutComputeOperations->push_back(Op);
	}

	{
		GLCSOperation Op;
		Op.Type = GLCSOperationType::WriteToOutput;
		OutComputeOperations->push_back(Op);
	}
}

static const char SwizzleStr[] = "xyzw";

void ConvertCSOperationsToComputeShaderSource(const std::vector<GLCSOperation>& ComputeOperations, std::string* OutString)
{
	(*OutString) = "";
	(*OutString) += "#version 430\n";
	(*OutString) += "layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;\n";
	(*OutString) += "layout(rgba32f, binding = 0) readonly uniform image2D inImg;\n";
	(*OutString) += "layout(rgba32f, binding = 1) coherent uniform image2D outImg;\n";
	(*OutString) += "\n";


	(*OutString) += "ivec2 getLocalSafeUVs(vec2 rawCoords) {\n";
	(*OutString) += "\tivec2 local_coords = ivec2(floor(fract(rawCoords.x) * gl_WorkGroupSize.x), floor(fract(rawCoords.y) * gl_WorkGroupSize.y));\n";
	(*OutString) += "\tivec2 work_group_start_coords = ivec2(gl_GlobalInvocationID.x - gl_LocalInvocationID.x, gl_GlobalInvocationID.y - gl_LocalInvocationID.y);\n";
	(*OutString) += "\treturn work_group_start_coords + local_coords;\n";
	(*OutString) += "}\n";
	
	(*OutString) += "vec4 sampleFromInputImg(vec2 rawCoords) {\n";
	(*OutString) += "\tivec2 imgSize = imageSize(inImg);\n";
	(*OutString) += "\tivec2 coords = ivec2(floor(fract(rawCoords.x) * imgSize.x), floor(fract(rawCoords.y) * imgSize.y));\n";
	(*OutString) += "\treturn imageLoad(inImg, coords);\n";
	(*OutString) += "}\n";

	(*OutString) += "void main() {\n";

	(*OutString) += "\tivec2 pxCoords = ivec2(gl_GlobalInvocationID.xy);\n";
	(*OutString) += "\tvec4 scratch = imageLoad(inImg, pxCoords);\n";

	auto ConvertCSValuetoSourceString = [](const GLCSValue& Value)
	{
		if (Value.Type == GLCSValueType::ReadFromInput)
		{
			// Hmmm...we want this to be based on scratch, but...we need some way of ensuring bounds
			// Maybe wrap or clamp it? Maybe make a helper function in the generated source?
			return std::string("sampleFromInputImg(scratch.") + SwizzleStr[Value.Index1] + SwizzleStr[Value.Index2] + ")";
		}
		else if (Value.Type == GLCSValueType::ReadFromOutput)
		{
			//return std::string("sampleFromImg(outImg, ivec2(255 - pxCoords.x, pxCoords.y))");
			return std::string("imageLoad(outImg, getLocalSafeUVs(scratch.") + SwizzleStr[Value.Index1] + SwizzleStr[Value.Index2] + "))";
		}
		else if (Value.Type == GLCSValueType::ReadFromOutputFixed)
		{
			return std::string("imageLoad(outImg, pxCoords)");
		}
		else if (Value.Type == GLCSValueType::Scratch)
		{
			return std::string("scratch");
		}
		else if (Value.Type == GLCSValueType::VecLiteral)
		{
			char Buffer[256] = {};
			snprintf(Buffer, sizeof(Buffer), "vec4(%.25f, %.25f, %.25f, %.25f)", Value.LiteralValue.x, Value.LiteralValue.y, Value.LiteralValue.z, Value.LiteralValue.w);
			return std::string(Buffer);
		}
		else
		{
			assert(false);
			return std::string("");
		}
	};

	// TODO:
	//std::vector<bool> ImageWrittenToInConditional;

	for (const auto& Op : ComputeOperations)
	{
		if (Op.Type == GLCSOperationType::Add)
		{
			(*OutString) += "\tscratch = " + ConvertCSValuetoSourceString(Op.Value1) + " + " + ConvertCSValuetoSourceString(Op.Value2) + ";\n";
		}
		else if (Op.Type == GLCSOperationType::Mul)
		{
			(*OutString) += "\tscratch = " + ConvertCSValuetoSourceString(Op.Value1) + " * " + ConvertCSValuetoSourceString(Op.Value2) + ";\n";
		}
		else if (Op.Type == GLCSOperationType::Sub)
		{
			(*OutString) += "\tscratch = " + ConvertCSValuetoSourceString(Op.Value1) + " - " + ConvertCSValuetoSourceString(Op.Value2) + ";\n";
		}
		else if (Op.Type == GLCSOperationType::Mov)
		{
			// Only really used at the end (and maybe the begining, idk)
			(*OutString) += "\tscratch = " + ConvertCSValuetoSourceString(Op.Value1) + ";\n";
		}
		else if (Op.Type == GLCSOperationType::BeginIf)
		{
			(*OutString) += "\tif (" + ConvertCSValuetoSourceString(Op.Value1) + "." + SwizzleStr[Op.IfStmt.ComponentIndex1]
				+ " < " + ConvertCSValuetoSourceString(Op.Value2) + "." + SwizzleStr[Op.IfStmt.ComponentIndex2] + ") {\n";
		}
		else if (Op.Type == GLCSOperationType::EndIf)
		{
			(*OutString) += "\t}\n";
			// TODO: May need to insert barriers here as well, if output was written to w/in the if statement
			//(*OutString) += "\tmemoryBarrier();\n";
			//(*OutString) += "\tbarrier();\n";
		}
		else if (Op.Type == GLCSOperationType::WriteToOutput)
		{
			(*OutString) += "\timageStore(outImg, pxCoords, scratch);\n";
			(*OutString) += "\tmemoryBarrier();\n";
			(*OutString) += "\tbarrier();\n";
			// TODO: Barriers
		}
	}

	(*OutString) += "}\n";
}


void CreateInputOutputImages(GLuint* OutInputImageID, GLuint* OutOutputImageID, GLuint* OutEmulatedOutputImageID, GLuint* OutEmulatedVsActualDiffID, int ImageWidth, int ImageHeight)
{
	GLuint InputImageID;
	glGenTextures(1, &InputImageID);
	glBindTexture(GL_TEXTURE_2D, InputImageID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	*OutInputImageID = InputImageID;

	GLuint OutputImageID;
	glGenTextures(1, &OutputImageID);
	glBindTexture(GL_TEXTURE_2D, OutputImageID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

	*OutOutputImageID = OutputImageID;

	GLuint EmulatedOutputImageID;
	glGenTextures(1, &EmulatedOutputImageID);
	glBindTexture(GL_TEXTURE_2D, EmulatedOutputImageID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

	*OutEmulatedOutputImageID = EmulatedOutputImageID;
;
	GLuint EmulatedVsActualDiffID;
	glGenTextures(1, &EmulatedVsActualDiffID);
	glBindTexture(GL_TEXTURE_2D, EmulatedVsActualDiffID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

	*OutEmulatedVsActualDiffID = EmulatedVsActualDiffID;
}

void WriteRandomBytesToInputImage(GLuint InputImageID, int ImageWidth, int ImageHeight, std::mt19937_64* RNGState, unsigned char* ImgBufferCPU)
{
	//glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, InputImageID);

	int NumBytes = ImageWidth * ImageHeight * 4;
	for (int i = 0; i < NumBytes; i++)
	{
		ImgBufferCPU[i] = (unsigned char)((*RNGState)() & 0xFF);
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, ImgBufferCPU);
}

// TODO: CS Metadata like local group size,
// and I guess image size
void RunComputeShaderSource(const std::string& ShaderSource, GLuint InputImageID, GLuint OutputImageID, int ImageWidth, int ImageHeight)
{
	GLuint ShaderObj = glCreateShader(GL_COMPUTE_SHADER);
	const char* ShaderSourcePtr = ShaderSource.c_str();
	glShaderSource(ShaderObj, 1, &ShaderSourcePtr, NULL);
	glCompileShader(ShaderObj);

	GLint CompileSuccess;
	glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &CompileSuccess);
	if (CompileSuccess == 0) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(ShaderObj, sizeof(infoLog), NULL, infoLog);
		OutputDebugStringA("Error compiling shader:\n");
		OutputDebugStringA(infoLog);
		OutputDebugStringA("\n");
		return;
	}

	GLuint ProgramObj = glCreateProgram();
	glAttachShader(ProgramObj, ShaderObj);
	glLinkProgram(ProgramObj);

	GLint LinkSuccess;
	glGetProgramiv(ProgramObj, GL_LINK_STATUS, &LinkSuccess);

	if (LinkSuccess == 0) {
		GLchar errorLog[1024];
		glGetProgramInfoLog(ProgramObj, sizeof(errorLog), NULL, errorLog);
		OutputDebugStringA("Error linking shader program:\n");
		OutputDebugStringA(errorLog);
		OutputDebugStringA("\n");
	}

	glUseProgram(ProgramObj);

	glBindImageTexture(0, InputImageID, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

	glBindImageTexture(1, OutputImageID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

	glDispatchCompute((GLuint)ImageWidth / 4, (GLuint)ImageHeight / 4, 1);

	

	// Barrier
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, InputImageID);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, OutputImageID);
}

struct EmulatedGLCSThreadState
{
	Vec4f Scratch;

	// If this is 0, the thread is active
	// If this is >0, the thread is inactive
	// Every time we encounter a begin if statement not taken, or an if statement while inactive, it's incremented
	// Every time it's >0 and encounters an end if, it's decremented
	int DisabledIfStack = 0;
};

// TODO: Now misleading, but...Idk how GL_NEAREST works
inline int RoundToNearest(float f)
{
	return (int)f;// (f + 0.5f);
}

inline float FloatFrac(float f)
{
	return f - floor(f);
	//double Dummy = 0.0f;
	//float FracPart = modf(f, &Dummy);
	//
	//if (FracPart < 0)
	//{
	//	assert(FracPart > -1.0f);
	//	return FracPart + 1.0f;
	//}
	//
	//return FracPart;
}

void RunEmulatedComputeShader(const std::vector<GLCSOperation>& OPs, const unsigned char* InputBuffer, float* OutputBuffer, int ImgWidth, int ImgHeight)
{
	// TODO: Params
	int LocalSizeX = 4;
	int LocalSizeY = 4;

	std::vector<EmulatedGLCSThreadState> ThreadStates;
	ThreadStates.resize(ImgWidth * ImgHeight);

	auto ReadFromOutputImageFixed = [&](int pxX, int pxY)
	{
		int Index = pxY * ImgWidth + pxX;
		Vec4f ret;
		for (int k = 0; k < 4; k++)
		{
			ret.xyzw[k] = OutputBuffer[Index * 4 + k];
		}
		return ret;
	};

	auto WriteOutputImageFixed = [&](Vec4f Val, int pxX, int pxY)
	{
		int Index = pxY * ImgWidth + pxX;
		for (int k = 0; k < 4; k++)
		{
			OutputBuffer[Index * 4 + k] = Val.xyzw[k];
		}
	};

	auto ReadFromOutputImage = [&](int pxX, int pxY, float X, float Y)
	{
		// TODO: Infinity, NaN, or just really big numbers
		int Xi = (pxX / LocalSizeX * LocalSizeX) + RoundToNearest(FloatFrac(X) * LocalSizeX);
		int Yi = (pxY / LocalSizeY * LocalSizeY) + RoundToNearest(FloatFrac(Y) * LocalSizeY);
		assert(Xi >= 0);
		assert(Yi >= 0);

		// Floating point occasionally means we hit the literal edge case,
		// in those cases just saturate
		assert(Xi <= ImgWidth);
		assert(Yi <= ImgHeight);

		if (Xi >= ImgWidth) { Xi = ImgWidth - 1; }
		if (Yi >= ImgHeight) { Yi = ImgHeight - 1; }

		int Index = Yi * ImgWidth + Xi;
		Vec4f ret;
		for (int k = 0; k < 4; k++)
		{
			ret.xyzw[k] = OutputBuffer[Index * 4 + k];
		}

		return ret;
	};

	auto ReadFromInputImage = [&](float X, float Y)
	{
		// TODO: Infinity, NaN, or just really big numbers
		int Xi = RoundToNearest(FloatFrac(X) * ImgWidth);
		int Yi = RoundToNearest(FloatFrac(Y) * ImgHeight);
		assert(Xi >= 0);
		assert(Yi >= 0);

		// Floating point occasionally means we hit the literal edge case,
		// in those cases just saturate
		assert(Xi <= ImgWidth);
		assert(Yi <= ImgHeight);

		if (Xi >= ImgWidth) { Xi = ImgWidth - 1; }
		if (Yi >= ImgHeight) { Yi = ImgHeight - 1; }

		int Index = Yi * ImgWidth + Xi;
		Vec4f ret;
		for (int k = 0; k < 4; k++)
		{
			ret.xyzw[k] = (double)(InputBuffer[Index * 4 + k]) / 255.0;
		}

		return ret;
	};

	auto EvaluateCSValue = [&](const GLCSValue& Value, int pxX, int pxY)
	{
		int Index = pxY * ImgWidth + pxX;
		EmulatedGLCSThreadState* ThreadState = &ThreadStates[Index];

		if (Value.Type == GLCSValueType::ReadFromInput)
		{
			// sample from round(frac(scratch.QQ) * imageWH)
			return ReadFromInputImage(ThreadState->Scratch.xyzw[Value.Index1], ThreadState->Scratch.xyzw[Value.Index2]);
		}
		else if (Value.Type == GLCSValueType::ReadFromOutput)
		{
			// GLSL
			// ivec2 local_coords = ivec2(floor(fract(rawCoords.x) * gl_WorkGroupSize.x), floor(fract(rawCoords.y) * gl_WorkGroupSize.y));
			// ivec2 work_group_start_coords = ivec2(gl_GlobalInvocationID.x - gl_LocalInvocationID.x, gl_GlobalInvocationID.y - gl_LocalInvocationID.y);
			// return work_group_start_coords + local_coords;

			return ReadFromOutputImage(pxX, pxY, ThreadState->Scratch.xyzw[Value.Index1], ThreadState->Scratch.xyzw[Value.Index2]);
		}
		else if (Value.Type == GLCSValueType::ReadFromOutputFixed)
		{
			return ReadFromOutputImageFixed(pxX, pxY);
		}
		else if (Value.Type == GLCSValueType::Scratch)
		{
			return ThreadState->Scratch;
		}
		else if (Value.Type == GLCSValueType::VecLiteral)
		{
			return Value.LiteralValue;
		}
		else
		{
			assert(false);
			return Vec4f();
		}
	};

	for (int j = 0; j < ImgHeight; j++)
	{
		for (int i = 0; i < ImgWidth; i++)
		{
			int Index = j * ImgWidth + i;
			for (int k = 0; k < 4; k++)
			{
				ThreadStates[Index].Scratch.xyzw[k] = (double)(InputBuffer[Index * 4 + k]) / 255.0;
			}
		}
	}


	for (const auto& OP : OPs)
	{
		for (int j = 0; j < ImgHeight; j++)
		{
			for (int i = 0; i < ImgWidth; i++)
			{
				int Index = j * ImgWidth + i;
				
				if (ThreadStates[Index].DisabledIfStack > 0)
				{
					if (OP.Type == GLCSOperationType::BeginIf)
					{
						ThreadStates[Index].DisabledIfStack++;
					}
					else if (OP.Type == GLCSOperationType::EndIf)
					{
						ThreadStates[Index].DisabledIfStack--;
					}
					else
					{
						// Do nothing, this thread did not take the branch we're currently emulating
					}
				}
				else
				{
					if (OP.Type == GLCSOperationType::Add)
					{
						Vec4f Val1 = EvaluateCSValue(OP.Value1, i, j);
						Vec4f Val2 = EvaluateCSValue(OP.Value2, i, j);
						ThreadStates[Index].Scratch = Val1 + Val2;
					}
					else if (OP.Type == GLCSOperationType::Mul)
					{
						Vec4f Val1 = EvaluateCSValue(OP.Value1, i, j);
						Vec4f Val2 = EvaluateCSValue(OP.Value2, i, j);
						ThreadStates[Index].Scratch = Val1 * Val2;
					}
					else if (OP.Type == GLCSOperationType::Sub)
					{
						Vec4f Val1 = EvaluateCSValue(OP.Value1, i, j);
						Vec4f Val2 = EvaluateCSValue(OP.Value2, i, j);
						ThreadStates[Index].Scratch = Val1 - Val2;
					}
					else if (OP.Type == GLCSOperationType::Mov)
					{
						Vec4f Val1 = EvaluateCSValue(OP.Value1, i, j);
						ThreadStates[Index].Scratch = Val1;
					}
					else if (OP.Type == GLCSOperationType::BeginIf)
					{
						// If the if statement holds, do nothing
						// If it does not hold, increment DisabledIfStack so we do not execute the following instructions
						Vec4f Val1 = EvaluateCSValue(OP.Value1, i, j);
						Vec4f Val2 = EvaluateCSValue(OP.Value2, i, j);
						float lhs = Val1.xyzw[OP.IfStmt.ComponentIndex1];
						float rhs = Val2.xyzw[OP.IfStmt.ComponentIndex2];
						if (lhs < rhs)
						{
							// Branch is taken, do nothing
						}
						else
						{
							ThreadStates[Index].DisabledIfStack++;
						}
					}
					else if (OP.Type == GLCSOperationType::EndIf)
					{
						// Do not decrement DisabledIfStack, since we are not in a disabled branch
					}
					else if (OP.Type == GLCSOperationType::WriteToOutput)
					{
						WriteOutputImageFixed(ThreadStates[Index].Scratch, i, j);
					}
					else
					{
						assert(false);
					}
				}
			}
		}
	}
}


void UploadEmulatedComputeShaderOutput(const float* EmulatedOutputImgBufferCPU, GLuint EmulatedOutputImageID, int ImageWidth, int ImageHeight)
{
	glBindTexture(GL_TEXTURE_2D, EmulatedOutputImageID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_FLOAT, EmulatedOutputImgBufferCPU);
}


//void glGetTexImage(GLenum target,
//	GLint level,
//	GLenum format,
//	GLenum type,
//	void* pixels);



void ComputeEmulatedActualDiff(const float* EmulatedOutputImgBufferCPU, float* OutputImgBufferCPU, float* EmulatedVsActualDiffBufferCPU, GLuint OutputImageID, GLuint EmulatedVsActualDiffID, int ImageWidth, int ImageHeight)
{
	int NumPixels = ImageWidth * ImageHeight;

	glBindTexture(GL_TEXTURE_2D, OutputImageID);
	glGetTextureImage(OutputImageID, 0, GL_RGBA, GL_FLOAT, NumPixels * sizeof(float) * 4, OutputImgBufferCPU);

	for (int i = 0; i < NumPixels * 4; i++)
	{
		EmulatedVsActualDiffBufferCPU[i] = abs(EmulatedOutputImgBufferCPU[i] - OutputImgBufferCPU[i]);
	}

	glBindTexture(GL_TEXTURE_2D, EmulatedVsActualDiffID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ImageWidth, ImageHeight, 0, GL_RGBA, GL_FLOAT, EmulatedVsActualDiffBufferCPU);
}

void ComputeDiffStats(const float* EmulatedVsActualDiffBufferCPU, DiffStats* DS, int ImageWidth, int ImageHeight)
{
	DS->TotalPixelErr = 0;
	DS->TotalDifferentPixels = 0;

	int NumPixels = ImageWidth * ImageHeight;
	for (int i = 0; i < NumPixels; i++)
	{
		Vec4f PixelDiff;
		for (int k = 0; k < 4; k++)
		{
			PixelDiff.xyzw[k] = EmulatedVsActualDiffBufferCPU[i * 4 + k];
		}

		float DiffMag = PixelDiff.Magnitude();
		DS->TotalPixelErr += DiffMag;
		if (DiffMag > 0.00001f)
		{
			DS->TotalDifferentPixels++;
		}
	}

	DS->AvgPixelErr = DS->TotalPixelErr / NumPixels;
}

