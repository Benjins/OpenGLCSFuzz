// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

#include <windows.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <stdio.h>

#include "gl_ext_api.h"

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void MainAppUpdate();

int main(int, char**)
{
	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
	// GL ES 2.0 + GLSL 100
	const char* glsl_version = "#version 100";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	// Create window with graphics context
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
	if (window == NULL)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
	bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
	bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
	bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
	bool err = gladLoadGL(glfwGetProcAddress) == 0; // glad2 recommend using the windowing library loader instead of the (optionally) bundled one.
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
	bool err = false;
	glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
	bool err = false;
	glbinding::initialize([](const char* name) { return (glbinding::ProcAddress)glfwGetProcAddress(name); });
#else
	InitGlExts();
	bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}

	{
		char Buffer[2048] = {};
		snprintf(Buffer, sizeof(Buffer), "GL version: %s\nGL vendor: %s\nGL Renderer: %s\n", glGetString(GL_VERSION), glGetString(GL_VENDOR), glGetString(GL_RENDERER));
		OutputDebugStringA(Buffer);
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'docs/FONTS.md' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);

	// Our state
	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	//int work_grp_cnt[3];
	//
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);



	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		MainAppUpdate();

		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}


struct DiffStats
{
	float AvgPixelErr = 0.0f;
	float TotalPixelErr = 0.0f;
	int TotalDifferentPixels = 0;
	float TotalPixelErrByChannel[4] = {};
};

// TODO: Header-ify
#include "gl_cs_gen.cpp"

#include <unordered_set>

struct AppContext
{
	std::string G_SourceBufferForDisplay = "";

	uint64_t RNGSeed = 0;

	bool bHasMadeTextures = false;
	GLuint InputImageID;
	GLuint OutputImageID;
	GLuint EmulatedOutputImageID;
	GLuint EmulatedVsActualDiffID;

	unsigned char* InputImgBufferCPU = nullptr;

	float* OutputImgBufferCPU = nullptr;

	float* EmulatedOutputImgBufferCPU = nullptr;

	float* EmulatedVsActualDiffBufferCPU = nullptr;

	bool ContinuousFuzz = false;

	bool ReRunOnNextFrame = false;

	std::vector<uint64_t> BadRNGSeeds;
	std::unordered_set<uint64_t> BadRNGSeedsSet;

	int NumCasesDone = 0;
	int NumCasesDifferent = 0;

	DiffStats DiffStats;
};

AppContext AC;

#include <random>

void MainAppUpdate() {
	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (!AC.bHasMadeTextures)
	{
		CreateInputOutputImages(&AC.InputImageID, &AC.OutputImageID, &AC.EmulatedOutputImageID, &AC.EmulatedVsActualDiffID, 256, 256);
		AC.InputImgBufferCPU = (unsigned char*)malloc(256 * 256 * 4);
		AC.OutputImgBufferCPU = (float*)malloc(256 * 256 * 4 * sizeof(float));
		AC.EmulatedOutputImgBufferCPU = (float*)malloc(256 * 256 * 4 * sizeof(float));
		AC.EmulatedVsActualDiffBufferCPU = (float*)malloc(256 * 256 * 4 * sizeof(float));
		AC.bHasMadeTextures = true;
	}

	// Show images of the results, the code (maybe), and stats (how many run, how fast, etc.), also highlight diffs, maybe save, pause, idk

	ImGui::InputScalar("Seed", ImGuiDataType_U64, &AC.RNGSeed);

	ImGui::Text("WARNING: Running 'Continuous Fuzz' may lead to rapidly changing noise patterns (probably ~20-30 Hz, but up to your monitor's refresh rate).");
	ImGui::Text("You can hit 'Single Fuzz' a couple times to see what the output is like");
	ImGui::Checkbox("Continuous Fuzz", &AC.ContinuousFuzz);

	bool bShouldRegenerateSource = false;

	if (AC.ContinuousFuzz)
	{
		bShouldRegenerateSource = true;

		std::random_device rd;
		AC.RNGSeed = rd();
		AC.RNGSeed <<= 30;
		AC.RNGSeed += rd();
	}

	if (ImGui::Button("Single Fuzz"))
	{
		bShouldRegenerateSource = true;

		std::random_device rd;
		AC.RNGSeed = rd();
		AC.RNGSeed <<= 30;
		AC.RNGSeed += rd();
	}

	if (AC.ReRunOnNextFrame)
	{
		bShouldRegenerateSource = true;
	}

	if (ImGui::Button("Regenerate Source"))
	{
		bShouldRegenerateSource = true;
	}

	if (bShouldRegenerateSource)
	{
		std::mt19937_64 MT64(AC.RNGSeed);
		std::vector<GLCSOperation> Ops;
		GenerateGLCSOperations(&MT64, &Ops);
		ConvertCSOperationsToComputeShaderSource(Ops, &AC.G_SourceBufferForDisplay);

		WriteRandomBytesToInputImage(AC.InputImageID, 256, 256, &MT64, AC.InputImgBufferCPU);

		RunComputeShaderSource(AC.G_SourceBufferForDisplay, AC.InputImageID, AC.OutputImageID, 256, 256);

		RunEmulatedComputeShader(Ops, AC.InputImgBufferCPU, AC.EmulatedOutputImgBufferCPU, 256, 256);

		UploadEmulatedComputeShaderOutput(AC.EmulatedOutputImgBufferCPU, AC.EmulatedOutputImageID, 256, 256);

		ComputeEmulatedActualDiff(AC.EmulatedOutputImgBufferCPU, AC.OutputImgBufferCPU, AC.EmulatedVsActualDiffBufferCPU, AC.OutputImageID, AC.EmulatedVsActualDiffID, 256, 256);

		ComputeDiffStats(AC.EmulatedVsActualDiffBufferCPU, &AC.DiffStats, 256, 256);

		AC.NumCasesDone++;

		if (AC.DiffStats.TotalDifferentPixels > 0)
		{
			if (AC.BadRNGSeedsSet.count(AC.RNGSeed) == 0)
			{
				AC.NumCasesDifferent++;

				AC.BadRNGSeeds.push_back(AC.RNGSeed);
				OutputDebugStringA("\n=====--------\n");

				char Buff[256] = {};
				snprintf(Buff, sizeof(Buff), "Found diff with RNG Seed: %llu\n", AC.RNGSeed);
				OutputDebugStringA(Buff);
				OutputDebugStringA("-----\n");

				OutputDebugStringA(AC.G_SourceBufferForDisplay.data());
				OutputDebugStringA("\n=====--------\n");

				AC.BadRNGSeedsSet.insert(AC.RNGSeed);
			}
		}
	}

	ImGui::Text("Input:  "); ImGui::SameLine();
	ImGui::Image((void*)(intptr_t)AC.InputImageID, ImVec2(256, 256));
	ImGui::SameLine();
	ImGui::Text("Output: "); ImGui::SameLine();
	ImGui::Image((void*)(intptr_t)AC.OutputImageID, ImVec2(256, 256));

	ImGui::Text("Emulated Output: "); ImGui::SameLine();
	ImGui::Image((void*)(intptr_t)AC.EmulatedOutputImageID, ImVec2(256, 256));
	ImGui::SameLine();
	ImGui::Text("Emulated vs Actual Diff: "); ImGui::SameLine();
	ImGui::Image((void*)(intptr_t)AC.EmulatedVsActualDiffID, ImVec2(256, 256));

	ImGui::Text("Pixel err: %.5f total (%d pixels), %.5f avg", AC.DiffStats.TotalPixelErr, AC.DiffStats.TotalDifferentPixels, AC.DiffStats.AvgPixelErr);
	//ImGui::Text("Pixel err: %.5f,%.5f,%.5f,%.5f total", AC.DiffStats.TotalPixelErrByChannel[0], AC.DiffStats.TotalPixelErrByChannel[1], AC.DiffStats.TotalPixelErrByChannel[2], AC.DiffStats.TotalPixelErrByChannel[3]);

	ImGui::Text("Number of cases: %4d", AC.NumCasesDone);
	ImGui::Text("Number of diff fails: %4d", AC.NumCasesDifferent);


	AC.ReRunOnNextFrame = false;

	if (ImGui::TreeNode("Bad Seeds"))
	{
		if (ImGui::Button("Print out current bad seeds"))
		{
			for (uint64_t BadSeed : AC.BadRNGSeeds)
			{
				char Buff[256] = {};
				snprintf(Buff, sizeof(Buff), "Bad seed: %llu\n", BadSeed);
				OutputDebugStringA(Buff);
			}
		}

		for (auto Iter = AC.BadRNGSeeds.begin(); Iter != AC.BadRNGSeeds.end(); )
		{
			uint64_t RNGSeed = *Iter;

			char Buff[256] = {};
			snprintf(Buff, sizeof(Buff), "REPRO##%llu", RNGSeed);
			if (ImGui::Button(Buff))
			{
				AC.RNGSeed = RNGSeed;
				AC.ReRunOnNextFrame = true;
			}

			ImGui::SameLine();
			ImGui::Text("Bad Seed: %llu", RNGSeed);
			ImGui::SameLine();

			snprintf(Buff, sizeof(Buff), "Remove##%llu", RNGSeed);
			if (ImGui::Button(Buff))
			{
				Iter = AC.BadRNGSeeds.erase(Iter);
			}
			else
			{
				Iter++;
			}
		}

		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Current Shader Source"))
	{
		ImGui::TextUnformatted(AC.G_SourceBufferForDisplay.data());

		ImGui::TreePop();
	}
}

