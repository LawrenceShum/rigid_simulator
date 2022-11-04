# include <glad/glad.h>
# include <GLFW/glfw3.h>
# include <iostream>
# include<vector>
# include<sstream>
# include<fstream>
# include <glm/glm.hpp>
# include <glm/gtc/matrix_transform.hpp>
# include <glm/gtc/type_ptr.hpp>
# include "shader_m.h"
# include "camera.h"
# include "cuda_header.cuh"
# include "global.h"
# include <cuda_gl_interop.h>
# include "rigid_body.cuh"

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
std::vector<float> bunny_vertices;
std::vector<int> bunny_indices;
// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;


void load_mesh(const std::string file_name, int& num_faces,int& num_vertices)
{
	std::ifstream in;
	in.open(file_name, std::ifstream::in);
	if (in.fail())
	{
		std::cout << "failed to open obj file" << std::endl;
		return;
	}
	std::string line;
	while (!in.eof())
	{
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "v "))
		{
			iss >> trash;
			for (int i = 0; i < 3; i++)
			{
				float x;
				iss >> x;
				bunny_vertices.push_back(1.0 * x);
			}
		}
		else if (!line.compare(0, 2, "f "))
		{
			int f, t, n;
			iss >> trash;
			iss >> f >> t >> n;
			bunny_indices.push_back(f - 1);
			bunny_indices.push_back(t - 1);
			bunny_indices.push_back(n - 1);
		}
	}
	in.close();
	num_vertices = bunny_vertices.size() / 3;
	num_faces = bunny_indices.size() / 3;
	std::cout << "total vertices = " << num_vertices << ", faces = " << num_faces << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(UP, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(DOWN, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);
}
// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		camera.ProcessMouseMovement(0.75 * xoffset, 0.75 * yoffset);
	}
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void move_bunny(double* vertices, int num, unsigned int& VBO)
{
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(double) * num, vertices, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main()
{
	Rigid_Body rigid(1.0, "model/stanford-bunny.obj");
	int num_vertices, num_faces;
	//load_mesh("model/stanford-bunny.obj", num_faces, num_vertices);
	num_vertices = rigid.access_num_vertices();
	num_faces = rigid.access_num_faces();
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Rigid Simulator", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "failed to initialize GLAD" << std::endl;
		return -1;
	}
	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glEnable(GL_DEPTH_TEST);

	Shader shaderProgram("src/vertex_shader.vs", "src/fragment_shader.fs");
	Shader shaderProgram_1("src/vertex_shader.vs", "src/fragment_shader_1.fs");
	Shader shaderProgram_2("src/vertex_shader_1.vs", "src/fragment_shader_2.fs");

	double* bunny = new double[num_vertices * 3];
	int* indices = new int[num_faces * 3];
	rigid.copy_to_opengl_buffer(bunny);
	rigid.access_indices(indices);
	
	double ground[12] = {
		-1, -0.5, -1,
		-1, -0.5, 1,
		1, -0.5, 1,
		1, -0.5, -1 };
	int ground_element[6] = {
		0, 1, 2,
		2, 3, 0
	};
	//unsigned int instanceVBO;
	//glGenBuffers(1, &instanceVBO);
	//glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(double) * 30, &translation[0], GL_STATIC_DRAW);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	//这是地面
	unsigned int VBO_GROUND, VAO_GROUND, EBO_GROUND;
	glGenVertexArrays(1, &VAO_GROUND);
	glGenBuffers(1, &VBO_GROUND);
	glGenBuffers(1, &EBO_GROUND);
	glBindVertexArray(VAO_GROUND);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_GROUND);
	glBufferData(GL_ARRAY_BUFFER, sizeof(double) * 4 * 3, &ground[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_GROUND);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 2 * 3, &ground_element[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), (void*)0);
	//解绑VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//解绑VAO
	glBindVertexArray(0);


	//这是兔子
	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(double)*num_vertices*3, bunny, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * num_faces * 3, indices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), (void*)0);
	//glEnableVertexAttribArray(1);
	//glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	//glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), (void*)0);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glVertexAttribDivisor(1, 1);

	//解绑VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//解绑VAO
	glBindVertexArray(0);

	while (!glfwWindowShouldClose(window))
	{
		// per-frame time logic
		// --------------------
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//变换
		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);
		view = camera.GetViewMatrix();
		projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

		//渲染兔子
		rigid.step_forward(0.001);
		rigid.copy_to_opengl_buffer(bunny);
		move_bunny(bunny, num_vertices * 3, VBO);
		shaderProgram.use();
		unsigned int modelLoc = glGetUniformLocation(shaderProgram.ID, "model");
		unsigned int viewLoc = glGetUniformLocation(shaderProgram.ID, "view");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
		shaderProgram.setMat4("projection", projection);
		shaderProgram.setMat4("view", view);
		shaderProgram.setMat4("model", model);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, num_faces * 3, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		shaderProgram_1.use();
		modelLoc = glGetUniformLocation(shaderProgram_1.ID, "model");
		viewLoc = glGetUniformLocation(shaderProgram_1.ID, "view");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
		shaderProgram_1.setMat4("projection", projection);
		shaderProgram_1.setMat4("view", view);
		shaderProgram_1.setMat4("model", model);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, num_faces * 3, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		shaderProgram_2.use();
		modelLoc = glGetUniformLocation(shaderProgram_2.ID, "model");
		viewLoc = glGetUniformLocation(shaderProgram_2.ID, "view");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
		shaderProgram_2.setMat4("projection", projection);
		shaderProgram_2.setMat4("view", view);
		shaderProgram_2.setMat4("model", model);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBindVertexArray(VAO_GROUND);
		glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	 
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	glfwTerminate();
	return 0;
}