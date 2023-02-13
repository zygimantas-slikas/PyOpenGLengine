#version 330 core

layout (location = 0) in vec2 in_texture_coordinates_0;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_position;

out vec2 uv_0;
out vec3 normal;
out vec3 fragment_position;

uniform mat4 camera_projection_matrix;
uniform mat4 view_matrix;

uniform vec3 scaling_factors;
uniform mat4 shear_transform;
uniform vec3 rotation_angles;
uniform vec3 translation_vector;

mat4 create_scaling(vec3 factors){
    mat4 result = mat4(0.0);
    result[0][0]=factors[0];
    result[1][1]=factors[1];
    result[2][2]=factors[2];
    result[3][3]=1.0;
    return result;
}

mat4 create_rotation(vec3 angles){
    mat4 x = mat4(0.0);
    mat4 y = mat4(0.0);
    mat4 z = mat4(0.0);
    for (int i = 0; i < 4; i++)
    {
        x[i][i] = 1.0;
        y[i][i] = 1.0;
        z[i][i] = 1.0;
    }
    x[1][1] = cos(angles[0]);
    x[1][2] = sin(angles[0]);
    x[2][1] = sin(angles[0])*-1;
    x[2][2] = cos(angles[0]);

    y[0][0] = cos(angles[1]);
    y[0][2] = sin(angles[1])*-1;
    y[2][0] = sin(angles[1]);
    y[2][2] = cos(angles[1]);

    z[0][0] = cos(angles[2]);
    z[0][1] = sin(angles[2]);
    z[1][0] = sin(angles[2])*-1;
    z[1][1] = cos(angles[2]);
    return x*y*z;
}

mat4 create_translation(vec3 direction){
    mat4 result = mat4(0.0);
    result[0][0]=1.0;
    result[1][1]=1.0;
    result[2][2]=1.0;
    result[3][3]=1.0;

    result[3][0]=direction[0];
    result[3][1]=direction[1];
    result[3][2]=direction[2];
    return result;
}

void main(){
    uv_0 = in_texture_coordinates_0;
    mat4 scaling = create_scaling(scaling_factors);
    mat4 rotation = create_rotation(rotation_angles);
    mat4 translation = create_translation(translation_vector);
    mat4 transformation = translation*rotation*shear_transform*scaling;

    fragment_position = vec3(transformation * vec4(in_position, 1.0));
    normal = mat3(transpose(inverse(transformation))) * normalize(in_normal);
    gl_Position = camera_projection_matrix * view_matrix * transformation * vec4(in_position, 1.0);
}