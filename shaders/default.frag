#version 330 core

layout (location =0) out vec4 fragColor;

in vec2 uv_0;
in vec3 normal;
in vec3 fragment_position;

struct Light{
    vec3 position;
    vec3 ambient_intensity;
    vec3 diffusion_intensity;
    vec3 specular_intensity;
};

uniform vec3 camera_position;
uniform Light light;
uniform sampler2D u_texture_0;

vec3 get_light(vec3 color){
    vec3 ambient = light.ambient_intensity;
    vec3 normal_1 = normalize(normal);

    vec3 light_direction = normalize(light.position - fragment_position);
    float light_alignment = max(0, dot(light_direction, normal_1));
    vec3 diffusion = light.diffusion_intensity * light_alignment;
    
    vec3 view_direction = normalize(camera_position - fragment_position);
    vec3 reflection_direction = reflect(-light_direction, normal_1);
    float spec = pow(max(dot(view_direction, reflection_direction), 0), 32);
    vec3 specular = spec * light.specular_intensity;

    return color * (ambient + diffusion + specular); 
}

void main(){
    float gamma = 2.2;
    vec3 color = texture(u_texture_0, uv_0).rgb;
    color = pow(color, vec3(gamma));
    color = get_light(color);
    color = pow(color, 1/vec3(gamma));
    fragColor = vec4(color, 1.0);
}