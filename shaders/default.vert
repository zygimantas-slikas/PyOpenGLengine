#version 330 core

layout (location = 0) in vec2 in_texture_coordinates_0;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_position;

out vec2 uv_0;
out vec3 normal;
out vec3 fragment_position;

uniform mat4 camera_projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

void main(){
    uv_0 = in_texture_coordinates_0;
    fragment_position = vec3(model_matrix * vec4(in_position, 1.0));
    normal = mat3(transpose(inverse(model_matrix))) * normalize(in_normal);
    gl_Position = camera_projection_matrix * view_matrix * model_matrix * vec4(in_position, 1.0);
}