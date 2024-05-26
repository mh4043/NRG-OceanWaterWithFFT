import GUI from './Scripts/muigui.module.js';
import {utils, mat4, vec4, vec3, vec2} from './Scripts/wgpu-matrix.module.js';

var simulationStarted = false;

// Ocean waves variables
var N; // The size of the Fourier grid (number of discrete points along x coord, number of columns)(from 16 to 2048 in powers of 2 (2048 = very detailed, typical 128 - 512))
var M; // The size of the Fourier grid (number of discrete points along z coord, number of rows)(from 16 to 2048 in powers of 2 (2048 = very detailed, typical 128 - 512))
// M, N determine resolution of the grid -> higher values, more detailed, finer grid
var Lx; // The physical dimension of the ocean, width (x) (in m or km)
var Lz; // The physical dimension of the ocean, length (z) (in m or km)
// Lx, Lz determine the scale of the ocean
// dx, dz => facet sizes, dx = Lx/M, dz = Lz/N (dx, dz larger than 2cm, smaller than V^2/g by substantial amount (10-1000))
var V; // Speed of wind (in m/s)
var omega; // Direction of wind (normalized value, horizontal (along x and z axis))
var l; // Minimum wave height (small wave cuttoff) (in m)
var A;/*0.000003;*/ // Numerical constant to increase or decrease wave height (10^-10 calmer conditions, 10^-5 rougher conditions)
var g; // Gravitational constant (m/s^2)
var lambda; // Wave choppiness

var num_of_points;

var time;
var previousTime;

// changing camera with spherical coordinates (radius, theta and phi) and not eye vector
var camera = {
    eye: [0, 0, 50], // position of the camera
    target: [0, 0, 0], // at what point is camera looking at
    up: [0, 1, 0], // up vector of the coordinate system
    radius: 500, // distance from the target
    theta: 0, // rotation on x-z axis
    phi: 40 * Math.PI/180, //rotation on y axis
    sensitivity: 0.01, // for orbiting
    moveSpeed: 10, // for moving
    zoomSpeed: 2, // for zooming
    // Variables to track mouse - moving camera
    isShiftPressed: false,
    // Variables to track mouse - orbital camera
    isDragging: false,
    lastMouseX: 0,
    lastMouseY: 0
};

var wave_height_field_final;
var wave_normal_field_final;
var initial_wave_height_field;
var initial_wave_height_field_conj;
var wave_height_field;

var indices; // for drawing triangles from points

// Transformation matrices
var viewMatrix; // world space -> camera space
var projectionMatrix; // camera space -> screen space

var lightPosition;

// For WebGPU
var canvas;
var device;
var context;
var observer;
var renderPipeline;
var bindGroup;
var renderPassDescriptor;
var vsUniformBuffer;
var fsUniformBuffer;
var vsUniformValues;
var fsUniformValues;
var viewMatrixUniform;
var projectionMatrixUniform;
var cameraPositionUniform;
var lightPositionUniform;

var vertexBufferPositions;
var vertexBufferNormals;
var indexBuffer;

var depthTexture;

var animationFrameId = null;

// Define a complex number
function Complex(real, imaginary) {
    this.real = real;
    this.imaginary = imaginary;
}
// Define a function to compute the addition of two complex numbers
Complex.prototype.add = function(other){
    return new Complex(
        this.real + other.real,
        this.imaginary + other.imaginary
    );
}
// Define a function to compute the product of two complex numbers
Complex.prototype.multiply = function(other) {
    return new Complex(
        this.real * other.real - this.imaginary * other.imaginary,
        this.real * other.imaginary + this.imaginary * other.real
    );
};
// Method to compute the conjugate of the complex number
Complex.prototype.conjugate = function() {
    return new Complex(this.real, -this.imaginary); // Negate the imaginary part
};


async function start(){
    initializeParameters();
    await initializeWebGPU();
    initializeTransformationMatrices();
    setEventHandlers();
    setSliderListeners();
    simulationStarted = true;
    main();
}

function main(){
    if(animationFrameId != null){
        cancelAnimationFrame(animationFrameId);
    }

    num_of_points = N * M;

    time = 0;
    previousTime = 0;

    initializeStartingArrays();
    initializeIndices();
    initializeVertexBuffers();
    
    computeInitialWaveHeightFields();

    animationFrameId = requestAnimationFrame(simulateOcean);
}

function initializeParameters(){
    //N = 512;
    N = 64;
    //N = 256;
    //M = 512;
    M = 64;
    //M = 256;
    Lx = 1000;
    //Lx = 100
    Lz = 1000;
    //Lz = 100
    V = 30;
    omega = vec2.fromValues(1, 1); //45°
    l = 0.1;
    A = 3e-7;
    g = 9.81;
    lambda = 1;

    lightPosition = vec3.fromValues(0, 20, 0);
}

function initializeStartingArrays(){
    // h(x,t)
    wave_height_field_final = new Array(num_of_points); // heights
    // delta_h(x,t)
    wave_normal_field_final = new Array(num_of_points); // normals

    // ~h0(k)
    initial_wave_height_field = new Array(num_of_points);
    // ~h*0(k)
    initial_wave_height_field_conj = new Array(num_of_points);
    // ~h(k,t)
    wave_height_field = new Array(num_of_points); // vsebuje pravilne pozicije točk (array of vec3 (x, y, z))
}

function initializeIndices(){
    // column-major (could be row-major - check)
    // 6 * (N - 1) * (M - 1)
    indices = [];
    for (let n = 0; n < N - 1; n++) {
        for (let m = 0; m < M - 1; m++) {
            var topLeft = getIndex(n, m, N);
            var topRight = getIndex(n, m + 1, N);
            var bottomLeft = getIndex(n + 1, m, N);
            var bottomRight = getIndex(n + 1, m + 1, N);
    
            // First triangle
            indices.push(topLeft, bottomLeft, topRight);
    
            // Second triangle
            indices.push(topRight, bottomLeft, bottomRight);
        }
    }
    indices = new Uint16Array(indices);
}
function getIndex(row, col, numRows) {
    return col * numRows + row;
}

function initializeTransformationMatrices(){
    // Initialize transformation matrices
    viewMatrix = mat4.create();
    projectionMatrix = mat4.create();

    // Initialize perspective parameters
    const fovy = utils.degToRad(90);
    const aspect = canvas.width / canvas.height;
    const near = 0.01;
    const far = 2000.0;

    mat4.perspective(fovy, aspect, near, far, projectionMatrix);
}

// Sets event handlers (visibilitychange, mouse, keyboard) (once per page load)
function setEventHandlers(){
    // Mouse event handlers (for camera orbiting, moving and zooming)
    canvas.addEventListener('mousedown', (event) => {
        if (event.button === 0) { // Left mouse button
            camera.isDragging = true;
            camera.lastMouseX = event.clientX;
            camera.lastMouseY = event.clientY;
        }
    });

    canvas.addEventListener('mousemove', (event) => {
        if (camera.isDragging) {
            const deltaX = event.clientX - camera.lastMouseX;
            const deltaY = event.clientY - camera.lastMouseY;

            if (camera.isShiftPressed) {
                // Camera movement
                const forward = vec3.create();
                vec3.subtract(camera.target, camera.eye, forward);
                vec3.normalize(forward, forward);

                const right = vec3.create();
                vec3.cross(forward, camera.up, right);
                vec3.normalize(right, right);

                vec3.addScaled(camera.eye, right, -deltaX * camera.sensitivity * camera.moveSpeed, camera.eye);
                vec3.addScaled(camera.eye, camera.up, deltaY * camera.sensitivity * camera.moveSpeed, camera.eye);
                vec3.addScaled(camera.target, right, -deltaX * camera.sensitivity * camera.moveSpeed, camera.target);
                vec3.addScaled(camera.target, camera.up, deltaY * camera.sensitivity * camera.moveSpeed, camera.target);
            } else {
                // Orbital rotation
                camera.theta += deltaX * camera.sensitivity;
                camera.phi += deltaY * camera.sensitivity;

                // Clamp phi to avoid flipping the camera upside down
                camera.phi = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, camera.phi));
            }

            camera.lastMouseX = event.clientX;
            camera.lastMouseY = event.clientY;
            updateViewMatrix();
        }
    });

    canvas.addEventListener('mouseup', (event) => {
        if (event.button === 0) {
            camera.isDragging = false;
        }
    });

    canvas.addEventListener('mouseleave', () => {
        camera.isDragging = false;
    });

    // Keyboard event handlers (for moving)
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Shift') {
            camera.isShiftPressed = true;
        }
    });

    document.addEventListener('keyup', (event) => {
        if (event.key === 'Shift') {
            camera.isShiftPressed = false;
        }
    });

    // Zoom event handler
    canvas.addEventListener('wheel', (event) => {
        const delta = Math.sign(event.deltaY);
        camera.radius += delta * camera.zoomSpeed;
        camera.radius = Math.max(1, camera.radius); // Prevent the camera from getting too close
    });
}
// Updates camera's position and view matrix (every frame)
function updateViewMatrix() {
    if (!camera.isShiftPressed) {
        camera.eye[0] = camera.target[0] + camera.radius * Math.sin(camera.theta) * Math.cos(camera.phi);
        camera.eye[1] = camera.target[1] + camera.radius * Math.sin(camera.phi);
        camera.eye[2] = camera.target[2] + camera.radius * Math.cos(camera.theta) * Math.cos(camera.phi);
    }
    mat4.lookAt(camera.eye, camera.target, camera.up, viewMatrix);
}

function setSliderListeners(){
    document.getElementById("sliderResolution").addEventListener('input', function() {
        var newResolution = N;
        switch(parseInt(this.value)){
            case 1:
                newResolution = 32;
                break;
            case 2:
                newResolution = 64;
                break;
            case 3:
                newResolution = 128;
                break;
            case 4:
                newResolution = 256;
                break;
            case 5:
                newResolution = 512;
                break;
        }
        N = newResolution;
        M = newResolution;
        document.getElementById("sliderResolutionValue").innerText = newResolution + "x" + newResolution;
        main();
    }, false);

    document.getElementById("sliderWaveHeight").addEventListener('input', function() {
        var newA = parseFloat(this.value) * Math.pow(10, -7);
        A = newA;
        main();
        document.getElementById("sliderWaveHeightValue").innerText = this.value + "e-7";
    }, false);

    document.getElementById("sliderWindSpeed").addEventListener('input', function() {
        var newSpeed = parseInt(this.value);
        V = newSpeed;
        main();
        document.getElementById("sliderWindSpeedValue").innerText = newSpeed + "m/s";
    }, false);

    document.getElementById("sliderWindDirection").addEventListener('input', function() {
        var dirDeg = parseInt(this.value);
        var dirRad = utils.degToRad(dirDeg);
        var newX = Math.cos(dirRad);
        var newZ = Math.sin(dirRad);
        omega = vec2.fromValues(newX, newZ);
        main();
        document.getElementById("sliderWindDirectionValue").innerText = dirDeg + "°";
    }, false);

    document.getElementById("sliderWaveCutoff").addEventListener('input', function() {
        var newCutoff = parseFloat(this.value);
        l = newCutoff;
        main();
        document.getElementById("sliderWaveCutoffValue").innerText = newCutoff + "m";
    }, false);

    document.getElementById("sliderWaveChoppiness").addEventListener('input', function() {
        lambda = parseFloat(this.value);
        document.getElementById("sliderWaveChoppinessValue").innerText = lambda;
    }, false);
}

async function initializeWebGPU(){
    // Check availability and get adapter and device
    if (!navigator.gpu) {
        fail('this browser does not support WebGPU');
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        fail('this browser supports webgpu but it appears disabled');
        return;
    }
    
    device = await adapter?.requestDevice();
    if (!device) {
        fail('need a browser that supports WebGPU');
        return;
    }

    // Get canvas, webgpu context and presentationFormat
    canvas = document.querySelector("#canv");
    context = canvas.getContext('webgpu');
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
    });

    // Create a shader module with vertex and fragment shader
    const shaderModule = device.createShaderModule({
        label: 'Rendering shader module',
        code:
        `
        struct VertexShaderUniforms {
            viewMatrix: mat4x4f,
            projectionMatrix: mat4x4f,
        };

        @group(0) @binding(0) var<uniform> vsUniforms: VertexShaderUniforms;

        struct VertexShaderInput {
            @location(0) position: vec3f,
            @location(1) normal: vec3f,
        };

        struct VertexShaderOutput {
            @builtin(position) position: vec4f,
            @location(0) normal: vec3f,
        };

        @vertex fn vertexShader(vertex: VertexShaderInput) -> VertexShaderOutput {
            var vsOut: VertexShaderOutput;
            vsOut.normal = (vsUniforms.viewMatrix * vec4f(vertex.normal, 0.0)).xyz;
            vsOut.position = vsUniforms.projectionMatrix * (vsUniforms.viewMatrix * vec4f(vertex.position, 1.0));
            return vsOut;
        }


        struct FragmentShaderUniforms {
            cameraPosition: vec3f,
            lightPosition: vec3f
        };

        @group(0) @binding(1) var<uniform> fsUniforms: FragmentShaderUniforms;

        fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }

        @fragment fn fragmentShader(vertex: VertexShaderOutput) -> @location(0) vec4f{
            //var a: vec4f = vec4f(fsUniforms.lightPosition, 1.0);
            //return vec4f(0.0, 0.0, 1.0, 1.0);

            var pos: vec3f = vertex.position.xyz;
            var normal: vec3f = normalize(vertex.normal);
            var viewDir: vec3f = normalize(fsUniforms.cameraPosition - pos);
            var lightDir: vec3f = normalize(fsUniforms.lightPosition - pos);
            var halfDir: vec3f = normalize(lightDir + viewDir);

            var NdotL: f32 = max(dot(normal, lightDir), 0.0);
            var NdotV: f32 = max(dot(normal, viewDir), 0.0);
            var NdotH: f32 = max(dot(normal, halfDir), 0.0);
            var VdotH: f32  = max(dot(viewDir, halfDir), 0.0);

            var F0: vec3f = vec3f(0.04);
            var F: vec3f = fresnelSchlick(max(dot(halfDir, viewDir), 0.0), F0);

            var roughness: f32 = 0.2; // Adjust based on the roughness of the ocean surface
            var D: f32 = pow(NdotH, (2.0 / (roughness * roughness)) - 2.0); // GGX Distribution

            var kS: vec3f = F;
            var kD: vec3f = vec3f(1.0) - kS;
            kD *= 1.0 - 0.0; // Assuming no metalness in water

            var diffuse: vec3f = kD * NdotL;
            var specular: vec3f = kS * D * NdotL;

            var reflectionDir: vec3f = reflect(-viewDir, normal);
            var refractionDir: vec3f = refract(-viewDir, normal, 1.0 / 1.33); // Assuming water refraction index is 1.33

            // Simple one-bounce tracing (reflection only)
            var reflectedColor: vec3f = vec3f(0.0, 0, 0.5); // Compute based on environment or skybox
            var refractedColor: vec3f = vec3f(0.0, 0.3, 0.7);; // Compute based on underwater color


            // Combining reflection and refraction
            var finalColor: vec3f = mix(reflectedColor, refractedColor, 0.5);

            return vec4f(finalColor + diffuse + specular, 1.0);
        }
          
        `
    });

    // Create render pipeline
    renderPipeline = device.createRenderPipeline({
        label: 'Render pipeline',
        layout: 'auto',
        vertex: {
            module: shaderModule,
            buffers: [
                // position
                {
                    arrayStride: 3 * 4, // 3 floats, 4 bytes each
                    attributes: [
                        {shaderLocation: 0, offset: 0, format: 'float32x3'},
                    ],
                },
                // normals
                {
                    arrayStride: 3 * 4, // 3 floats, 4 bytes each
                    attributes: [
                        {shaderLocation: 1, offset: 0, format: 'float32x3'},
                    ],
                },
            ],
        },
        fragment: {
            module: shaderModule,
            targets: [{ format: presentationFormat }],
        },
        /*
        // dont draw triangles that are facing back (cull => dont draw)
        primitive: {
            cullMode: 'back',
        },
        */
        // depth testing - correctly draw based on Z value (if Z value of new pixel is less than of drawn one => draw, else dont)
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
    });

    // Preparing for uniforms
    // Set sizes of uniform structs
    const vertUniformBufferSize = 2 * 16 * 4; // 2 mat4s * 16 floats per mat * 4 bytes per float
    const fragUniformBufferSize = 2 * 3 * 4 + 8; // 2 vec3 * 3 floats per vec3 * 4 bytes per float (might need to add padding), +8 = padding (min size is 32bytes)

    // Create uniform buffers
    vsUniformBuffer = device.createBuffer({
        size: vertUniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    fsUniformBuffer = device.createBuffer({
        size: fragUniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create uniform arrays and array views
    vsUniformValues = new Float32Array(2 * 16); // 2 mat4s
    viewMatrixUniform = vsUniformValues.subarray(0, 16);
    projectionMatrixUniform = vsUniformValues.subarray(16, 32);
    fsUniformValues = new Float32Array(2 * 3);  // 2 vec3fs
    cameraPositionUniform = fsUniformValues.subarray(0, 3);
    lightPositionUniform = fsUniformValues.subarray(3, 6);

    // Create bind group -> what resources the shader will use (getBindGroupLayout(0) = @group(0), binding: 0 = @binding(0))
    // Error because in fragment shader, FragmentShaderUniforms is currently not used, so it discards entry with fsUniformBuffer
    bindGroup = device.createBindGroup({
        label: 'bind group for shader module',
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: vsUniformBuffer } },
            { binding: 1, resource: { buffer: fsUniformBuffer } },
        ],
    });

    // Create render pass descriptor
    renderPassDescriptor = {
        colorAttachments: [
            {
                clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
        ],
        depthStencilAttachment: {
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        },
    };

    observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            const canvas = entry.target;
            const width = entry.contentBoxSize[0].inlineSize;
            const height = entry.contentBoxSize[0].blockSize;
            canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
            canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
            // re-render
            render();
        }
    });
    observer.observe(canvas);
}

function initializeVertexBuffers(){
    // Set buffers
    vertexBufferPositions = device.createBuffer({
        label: 'vertex buffer vertices',
        size: num_of_points * 4 * 3, // we have M * N points, which are floats (1 float = 4bytes) and 3 components (x, y, z)
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    vertexBufferNormals = device.createBuffer({
        label: 'vertex buffer normals',
        size: num_of_points * 4 * 3,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    indexBuffer = device.createBuffer({
        label: 'index buffer',
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indices);
}


function render(){
    // Get the current texture from the canvas context and
    // set it as the texture to render to.
    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    // UPDATE UNIFORM VALUES
    updateViewMatrix();

    mat4.copy(projectionMatrix, projectionMatrixUniform);
    mat4.copy(viewMatrix, viewMatrixUniform);
    vec3.copy(vec3.fromValues(camera.eye[0], camera.eye[1], camera.eye[2]), cameraPositionUniform);
    vec3.copy(lightPosition, lightPositionUniform);

    device.queue.writeBuffer(vsUniformBuffer, 0, vsUniformValues);
    device.queue.writeBuffer(fsUniformBuffer, 0, fsUniformValues);

    // If we don't have a depth texture OR if its size is different
    // from the canvasTexture when make a new depth texture
    if (!depthTexture ||
        depthTexture.width !== canvasTexture.width ||
        depthTexture.height !== canvasTexture.height) {
        if (depthTexture) {
            depthTexture.destroy();
        }
        depthTexture = device.createTexture({
            size: [canvasTexture.width, canvasTexture.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }
    renderPassDescriptor.depthStencilAttachment.view = depthTexture.createView();

    var wave_height_field_final_flat = new Float32Array(wave_height_field_final.flatMap((vec) => ([vec[0], vec[1], vec[2]])));
    var wave_normal_field_final_flat = new Float32Array(wave_normal_field_final.flatMap((vec) => ([vec[0], vec[1], vec[2]])));

    device.queue.writeBuffer(vertexBufferPositions, 0, wave_height_field_final_flat);
    device.queue.writeBuffer(vertexBufferNormals, 0, wave_normal_field_final_flat);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, vertexBufferPositions);
    passEncoder.setVertexBuffer(1, vertexBufferNormals);
    passEncoder.setIndexBuffer(indexBuffer, 'uint16');
    passEncoder.drawIndexed(indices.length);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

// FFTs
/*
math.fft(array of values), math.ifft(array of math.complex values) => official math.js library
or
transform(array of real values, array of imaginary values(zeros)), inverseTransform(array of real values, array of imaginary values) => lightweight fft library
with inverseTransform(), you have to scale the output real array (divite with the size of array)
*/

// (Equation 36) indexes can't be negative (-N/2 <= n < N/2), so we use indexes that are positive (0 >= n < N), and because of that
// we need to change the k(kx, kz) vector a bit, so instead of kx = 2*PI*n/Lx, we now have 2*PI*(n-N/2)/Lx so that its values remain the same
function vectorK(n, m){
    // original
    var deltaKx = 2 * Math.PI / Lx;
    var deltaKz = 2 * Math.PI / Lz;
    // changed
    var kx = (n - N / 2) * deltaKx;
    var kz = (m - M / 2) * deltaKz;
    return vec2.fromValues(kx, kz);
}

function randomGaussian(mean, stddev){
    // Box-Muller transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
    /*
    const u = 1 - Math.random(); // Converting [0,1) to (0,1]
    const v = Math.random();
    const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return z * stddev + mean; // Transform to the desired mean and standard deviation
    */
    var u1 = Math.random();
    var u2 = Math.random();
    var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z * stddev + mean; // Transform to the desired mean and standard deviation
}

// Equation 40
function phillips(/*vec2*/ k_vec){
    if(k_vec[0] == 0 && k_vec[0] == 0){
        return 0.0;
    }

    var L = Math.pow(V, 2) / g; // the largest possible waves arising from a continuous wind of speed V

    var k = vec2.length(k_vec); // magnitude/length of vector k
    var k_hat = vec2.fromValues(0, 0); // unit vector (normalized vector) in the direction of vector k
    vec2.normalize(k_vec, k_hat); 

    var omega_hat = vec2.fromValues(0, 0);
    vec2.normalize(omega, omega_hat);

    // suppress waves smaller than a small length l << L (+ add a multiplicative factor to spectrum equation)
    
    var temp_1 = A;
    var temp_2 = Math.exp(-1.0 / Math.pow(k * L, 2)) / Math.pow(k, 4);
    //var temp_3 = Math.pow(Math.abs(vec2.dot(k_hat, omega_hat)), 2);
    var temp_3 = Math.pow(vec2.dot(k_hat, omega_hat), 6); // 2 => surface less aligned with the wind direction
    var temp_4 = Math.exp(-Math.pow(k, 2) * Math.pow(l, 2)); // suppression of smaller waves factor (Equation 41)

    var phillipsVal = temp_1 * temp_2 * temp_3 * temp_4;

    return phillipsVal;
}

// Equation 42 (is calculated only once per parameter change - if no parameters change, then this function doesn't get called)
// we put all the initial values into an array
function initialWaveHeightField(/*vec2*/ k_vec){
    var random_real = randomGaussian(0, 1);
    var random_imaginary = randomGaussian(0, 1);

    var temp_1 = new Complex(1 / Math.sqrt(2), 0); // 1 / sqrt(2) == sqrt(0.5)
    var temp_2 = new Complex(random_real, random_imaginary);
    var temp_3 = new Complex(Math.sqrt(phillips(k_vec)), 0);

    var initialHeight = temp_2.multiply(temp_3).multiply(temp_1);

    return initialHeight;
}

function computeInitialWaveHeightFields(){
    // initialize initial_wave_height_field and initial_wave_height_field_conj for Equation 43, by Equation 42
    // with indexes we go from 0 to N or M instead of from -N/2 to N/2 or M
    // N -> number of columns, M -> number of rows
    for(var n = 0; n < N; ++n){
        for(var m = 0; m < M; ++m){
            // column-major ordering
            var index = m * N + n;
            var k_vec = vectorK(n, m);
            var k_vec_neg = vec2.fromValues(0, 0);
            vec2.negate(k_vec, k_vec_neg); // instead of conjugation, negation of vector k is used (it preserves the property as stated in the paper)
            // this only gets called once per startup or parameter change, otherwise we take values from these arrays
            var iwhf = initialWaveHeightField(k_vec);
            initial_wave_height_field[index] = iwhf;
            //var iwhfc = initialWaveHeightField(k_vec_neg);
            var iwhfc = initialWaveHeightField(k_vec).conjugate();
            initial_wave_height_field_conj[index] = iwhfc;
        }
    }
}

// Equation 43
function waveHeightField(n, m, t){
    var index = m * N + n;
    var k_vec = vectorK(n, m);
    var k = vec2.length(k_vec);
    var omega_k = Math.sqrt(g * k);

    // Euler formula: exp(ix)=cos(x)+isin(x)
    var temp_1_1 = new Complex(Math.cos(omega_k * t), Math.sin(omega_k * t));
    var temp_2_1 = new Complex(Math.cos(-omega_k * t), Math.sin(-omega_k * t));
    var temp_1 = initial_wave_height_field[index].multiply(temp_1_1);
    var temp_2 = initial_wave_height_field_conj[index].multiply(temp_2_1);
    var waveHeight = temp_1.add(temp_2);

    return waveHeight;
}

function simulateOcean(now){
    // Computing time between frames
    now *= 0.001; // convert to seconds
    const deltaTime = now - previousTime; // compute time since last frame
    const fps = 1 / deltaTime; // compute frames per second
    previousTime = now; // remember time for next frame

        // --- Setting variables ---

    // for slope vectors (normal), Equation 37
    var slope_x = new Array(num_of_points);
    var slope_z = new Array(num_of_points);

    // for wave choppiness (displacement), Equation 44
    var displacement_x = new Array(num_of_points);
    var displacement_z = new Array(num_of_points);

    // --- Calculating wave height field ---
    //console.time("Calculating wave height field");

    for(var n = 0; n < N; n++){
        for(var m = 0; m < M; m++){
            // column-major ordering (it jumps from 0 to N to 2N to 3N... (column-major), as opposed to 0, 1, 2, 3... (row-major))
            // first point at (0, 0), second point at (N, 0), third at (2N, 0) -> first column
            var index = m * N + n;

            var whf = waveHeightField(n, m, time);
            wave_height_field[index] = whf;

            var k_vec = vectorK(n, m);
            var k_length = vec2.length(k_vec);
            var k_normalized = k_vec;

            if(k_length != 0){
                vec2.normalize(k_vec, k_normalized);
            }

            // For slope vector x and z seperately for computing normals (Equation 37)
            var sx = new Complex(0, k_vec[0]).multiply(wave_height_field[index]);
            slope_x[index] = sx;
            var sz = new Complex(0, k_vec[1]).multiply(wave_height_field[index]);
            slope_z[index] = sz;

            // For displacement vector x and z seperately (Equation 44)
            var dx = new Complex(0, -k_normalized[0]).multiply(wave_height_field[index]);
            displacement_x[index] = dx;
            var dz = new Complex(0, -k_normalized[1]).multiply(wave_height_field[index]);
            displacement_z[index] = dz;
        }
    }

    //console.timeEnd("Calculating wave height field");

    // Creating and populating real and imaginary arrays for inverse FFT with inverseTransform(real, imag) from fft.js

    // Arrays to hold the real and imaginary parts
    let wave_height_field_real = new Float64Array(num_of_points);
    let wave_height_field_imag = new Float64Array(num_of_points);
    let slope_x_real = new Float64Array(num_of_points);
    let slope_x_imag = new Float64Array(num_of_points);
    let slope_z_real = new Float64Array(num_of_points);
    let slope_z_imag = new Float64Array(num_of_points);
    let displacement_x_real = new Float64Array(num_of_points);
    let displacement_x_imag = new Float64Array(num_of_points);
    let displacement_z_real = new Float64Array(num_of_points);
    let displacement_z_imag = new Float64Array(num_of_points);

    // Populate the real and imaginary arrays
    //console.time("Populating the real and imaginary arrays");

    for (var i = 0; i < N * M; i++) {
        wave_height_field_real[i] = (wave_height_field[i].real);
        wave_height_field_imag[i] = (wave_height_field[i].imaginary);
        slope_x_real[i] = (slope_x[i].real);
        slope_x_imag[i] = (slope_x[i].imaginary);
        slope_z_real[i] = (slope_z[i].real);
        slope_z_imag[i] = (slope_z[i].imaginary);
        displacement_x_real[i] = (displacement_x[i].real);
        displacement_x_imag[i] = (displacement_x[i].imaginary);
        displacement_z_real[i] = (displacement_z[i].real);
        displacement_z_imag[i] = (displacement_z[i].imaginary);
    }
    //console.timeEnd("Populating the real and imaginary arrays");

    // Calculating inverse FFT
    //console.time("Performing IFFTs");

    // Equations 36, 37, 44 (represents the IFFT from the frequency domain back to the spatial domain), whole function is done by inverse transform
    // results are written back to original arrays
    // ifft usually divides computed values by number of points, but in our equations, that is not used, and inverseTransform() doesn't divide either
    inverseTransform(wave_height_field_real, wave_height_field_imag);
    inverseTransform(slope_x_real, slope_x_imag);
    inverseTransform(slope_z_real, slope_z_imag);
    inverseTransform(displacement_x_real, displacement_x_imag);
    inverseTransform(displacement_z_real, displacement_z_imag);

    //console.timeEnd("Performing IFFTs");

    // Creating normals and heights and setting the final normal and height fields

    //console.time("Creating normals and heights");

    for(var n = 0; n < N; n++){
        for(var m = 0; m < M; m++){
            var index = m * N + n;
            var sign = 1;

            // flip the sign
            if((m + n) % 2){
                sign = -1;
            }

            // function to get the normal, which is perpendicular to the surface with slopes x and z
            // sign, which is alternating between 1 and -1 means, that we have an alternating pattern throughout the grid - waves
            var normal = vec3.fromValues(sign * slope_x_real[index], -1, sign * slope_z_real[index]);
            vec3.normalize(normal, normal);
            wave_normal_field_final[index] = normal;

            // Equation below 44 (x = lambda*D(x,t)), lambda is a scaling factor for importance of displacements
            // original: x = (nLx/N, mLz/M), ampak ker gremo z indeksi n (m) od 0 do N (M) in ne od -N/2 do N/2 (M) kot v članku (indeksi ne morejo biti negativni), je treba tudi x popraviti tako kot vektor k, potem pa dodamo še displacement z lamdbdo x*labda*disp
            var x_x = (n - N / 2) * Lx / N;
            var x_z = (m - M / 2) * Lz / M;
            var height = vec3.fromValues(x_x - sign * lambda * displacement_x_real[index], sign * wave_height_field_real[index], x_z - sign * lambda * displacement_z_real[index]);
            wave_height_field_final[index] = height;
        }
    }

    //console.timeEnd("Creating normals and heights");

    render();

    time += 0.15;
    animationFrameId = requestAnimationFrame(simulateOcean);
}


document.addEventListener("DOMContentLoaded", function() {

    const startButton = document.getElementById("btnStartSimulation");
    startButton.addEventListener('click', function(event){
        if(!simulationStarted){
            start();
        }else{
            return;
        }
    });

 });