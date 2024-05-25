
/*

- Celoten algoritem napisat na grafični kartici - tut merjenje višine (vertex ali fragment shader)
- WebGL nima compute shaderjev, zato je treba FFT spisat v fragment shaderju s pomočjo tekstur (dostopaš preko UV koordinat)
WebGPU pa ma compute shaderje in bi blu lažji tm zračunat FFT, sam je pa drugačn shader jezik

*/

// WebGL and rendering varables
var gl; // webgl canvas context
var canvas;
var indices; // for drawing triangles from points
var wave_height_field_final_flat; // flattened version of wave_height_field
var wave_normal_field_final_flat; // flattened version of wave_normal_field

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

var lambda;
var heightMax;
var heightMin;
var time;
var previousTime;

var num_of_points;
var wave_height_field_final;
var wave_normal_field_final;
var initial_wave_height_field;
var initial_wave_height_field_conj;
var wave_height_field;

// Initialize camera
var camera;

// Locations
// uniform locations
var viewMatrixLocation;
var projectionMatrixLocation;
var cameraPosition;
var lightPosition;
// attribute locations
var positionsLocation;
var normalsLocation;

// Buffers
var vertexBuffer;
var indexBuffer;
var normalBuffer;

// Transformation matrices
var viewMatrix; // world space -> camera space
var projectionMatrix; // camera space -> screen space
var viewProjectionMatrix; // world space -> screen space

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
                vec3.subtract(forward, camera.target, camera.eye);
                vec3.normalize(forward, forward);

                const right = vec3.create();
                vec3.cross(right, forward, camera.up);
                vec3.normalize(right, right);

                vec3.scaleAndAdd(camera.eye, camera.eye, right, -deltaX * camera.sensitivity * camera.moveSpeed);
                vec3.scaleAndAdd(camera.eye, camera.eye, camera.up, deltaY * camera.sensitivity * camera.moveSpeed);
                vec3.scaleAndAdd(camera.target, camera.target, right, -deltaX * camera.sensitivity * camera.moveSpeed);
                vec3.scaleAndAdd(camera.target, camera.target, camera.up, deltaY * camera.sensitivity * camera.moveSpeed);
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
    mat4.lookAt(viewMatrix, camera.eye, camera.target, camera.up);
}


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


function main(){
    initializeParameters();
    initializeStartingArrays();
    initializeWebGL();
    initializeIndices();
    setEventHandlers();
    computeInitialWaveHeightFields();
    
    requestAnimationFrame(simulateOcean);
}

function initializeParameters(){
    //N = 512;
    //N = 64;
    N = 256;
    //M = 512;
    //M = 64;
    M = 256;
    Lx = 1000;
    //Lx = 100
    Lz = 1000;
    //Lz = 100
    V = 30;
    omega = vec2.fromValues(1, 1);
    l = 0.1;
    A = 3e-7;
    g = 9.81;

    num_of_points = N * M;

    lambda = 1;
    heightMax = 0;
    heightMin = 0;
    time = 0;
    previousTime = 0;

    // changing camera with spherical coordinates (radius, theta and phi) and not eye vector
    camera = {
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

function initializeWebGL(){
    // Get a WebGL context
    canvas = document.querySelector("#canv");
    gl = canvas.getContext("webgl");
    if (!gl) {
        console.log("WebGL not supported");
        return;
    }

    // setup shaders and program
    var vertexShaderSource = document.getElementById("vertex-shader").text;
    var fragmentShaderSource = document.getElementById("fragment-shader").text;

    // - vertex shader
    var vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)){
        console.log(gl.getShaderInfoLog(vertexShader));
    }

    // - fragment shader
    var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)){
        console.log(gl.getShaderInfoLog(fragmentShader));
    }

    // - program
    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)){
        console.log(gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);

    webglUtils.resizeCanvasToDisplaySize(gl.canvas);

    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Clear the canvas.
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Attribute locations
    positionsLocation = gl.getAttribLocation(program, "aPosition");
    normalsLocation = gl.getAttribLocation(program, "aNormal");

    // Uniforms locations
    viewMatrixLocation = gl.getUniformLocation(program, "uViewMatrix");
    projectionMatrixLocation = gl.getUniformLocation(program, "uProjectionMatrix");
    cameraPosition = gl.getUniformLocation(program, "uCameraPosition");
    lightPosition = gl.getUniformLocation(program, "uLightPosition");

    // Enable vertex attribute
    gl.enableVertexAttribArray(positionsLocation);
    gl.enableVertexAttribArray(normalsLocation);

    // Create buffers
    vertexBuffer = gl.createBuffer();
    indexBuffer = gl.createBuffer();
    normalBuffer = gl.createBuffer();

    // Initialize transformation matrices
    viewMatrix = mat4.create();
    projectionMatrix = mat4.create();

    // Initialize perspective parameters
    const fovy = glMatrix.toRadian(90);
    const aspect = canvas.width / canvas.height;
    const near = 0.01;
    const far = 2000.0;
    // setup projection matrix
    //mat4.ortho(projectionMatrix, -2, 2, -2, 2, 0.1, 100); // if i want points to be orthogonal - same distances apart based on viewing
    mat4.perspective(projectionMatrix, fovy, aspect, near, far);

    // !! Naenkrat je lahko bindan samo en buffer, zato je treba vse inicializirat in nastavit prej preden zamenjamo buffer na drugo spremenljivko (https://stackoverflow.com/questions/36812755/webgl-rendering-fails-when-using-color-buffer) !!
}

function getIndex(row, col, numRows) {
    return col * numRows + row;
}
function initializeIndices(){
    // column-major (could be row-major - check)
    indices = [];
    for (let n = 0; n < N - 1; n++) {
        for (let m = 0; m < M - 1; m++) {
            /*
            let top_left = m * N + n;
            let top_right = top_left + 1;
            let bottom_left = (m + 1) * N + n;
            let bottom_right = bottom_left + 1;
            */
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
}

function computeInitialWaveHeightFields(){
    // initialize initial_wave_height_field and initial_wave_height_field_conj for Equation 43, by Equation 42
    // with indexes we go from 0 to N or M instead of from -N/2 to N/2 or M
    // N -> number of columns, M -> number of rows
    for(var n = 0; n < N; ++n){
        for(var m = 0; m < M; ++m){
            // column-major ordering
            var index = m * N + n;
            k_vec = vectorK(n, m);
            k_vec_neg = vec2.fromValues(0, 0);
            vec2.negate(k_vec_neg, k_vec); // instead of conjugation, negation of vector k is used (it preserves the property as stated in the paper)
            // this only gets called once per startup or parameter change, otherwise we take values from these arrays
            var iwhf = initialWaveHeightField(k_vec);
            initial_wave_height_field[index] = iwhf;
            //var iwhfc = initialWaveHeightField(k_vec_neg);
            var iwhfc = initialWaveHeightField(k_vec).conjugate();
            initial_wave_height_field_conj[index] = iwhfc;
        }
    }
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
    vec2.normalize(k_hat, k_vec); 

    var omega_hat = vec2.fromValues(0, 0);
    vec2.normalize(omega_hat, omega);

    // suppress waves smaller than a small length l << L (+ add a multiplicative factor to spectrum equation)
    
    var temp_1 = A;
    var temp_2 = Math.exp(-1.0 / Math.pow(k * L, 2)) / Math.pow(k, 4);
    //var temp_3 = Math.pow(Math.abs(vec2.dot(k_hat, omega_hat)), 2);
    var temp_3 = Math.pow(vec2.dot(k_hat, omega_hat), 2);
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
    console.time("Calculating wave height field");

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
                vec2.normalize(k_normalized, k_vec);
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

    console.timeEnd("Calculating wave height field");

    /*
    // mathjs.ifft()
    console.log("CONVERTING ARRAYS: ")
    let wave_height_field_math_1 = wave_height_field.map(c => math.complex(c.real, c.imaginary));
    let slope_x_math = slope_x.map(c => math.complex(c.real, c.imaginary));
    let slope_z_math = slope_z.map(c => math.complex(c.real, c.imaginary));
    let displacement_x_math = displacement_x.map(c => math.complex(c.real, c.imaginary));
    let displacement_z_math = displacement_z.map(c => math.complex(c.real, c.imaginary));

    console.log("PERFORMING IFFTs: ")
    wave_height_field_math = math.ifft(wave_height_field_math_1);
    slope_x_math = math.ifft(slope_x_math);
    slope_z_math = math.ifft(slope_z_math);
    displacement_x_math = math.ifft(displacement_x_math);
    displacement_z_math = math.ifft(displacement_z_math);
    console.log(wave_height_field_math);

    console.log("CONVERTING ARRAYS: ")
    //wave_height_field = wave_height_field_math.map(c => new Complex(c.real, c.imaginary));
    slope_x = slope_x_math.map(c => new Complex(c.real, c.imaginary));
    slope_z = slope_z_math.map(c => new Complex(c.real, c.imaginary));
    displacement_x = displacement_x_math.map(c => new Complex(c.real, c.imaginary));
    displacement_z = displacement_z_math.map(c => new Complex(c.real, c.imaginary));
    */

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
    console.time("Populating the real and imaginary arrays");

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
    console.timeEnd("Populating the real and imaginary arrays");

    // Calculating inverse FFT
    console.time("Performing IFFTs");

    // Equations 36, 37, 44 (represents the IFFT from the frequency domain back to the spatial domain), whole function is done by inverse transform
    // results are written back to original arrays
    // ifft usually divides computed values by number of points, but in our equations, that is not used, and inverseTransform() doesn't divide either
    inverseTransform(wave_height_field_real, wave_height_field_imag);
    inverseTransform(slope_x_real, slope_x_imag);
    inverseTransform(slope_z_real, slope_z_imag);
    inverseTransform(displacement_x_real, displacement_x_imag);
    inverseTransform(displacement_z_real, displacement_z_imag);

    console.timeEnd("Performing IFFTs");

    // Creating normals and heights and setting the final normal and height fields

    console.time("Creating normals and heights");

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
            x_x = (n - N / 2) * Lx / N;
            x_z = (m - M / 2) * Lz / M;
            var height = vec3.fromValues(x_x - sign * lambda * displacement_x_real[index], sign * wave_height_field_real[index], x_z - sign * lambda * displacement_z_real[index]);
            wave_height_field_final[index] = height;
        }
    }

    console.timeEnd("Creating normals and heights");

    // Calculating min and max heights

    for (var i = 0; i < N; i++){
        for (var j = 0; j < M; j++)
        {	
            var index = j * N + i;

            if (wave_height_field_final[index][1] > heightMax) 
                heightMax = wave_height_field_final[index][1];
            else if (wave_height_field_final[index][1] < heightMin) 
                heightMin = wave_height_field_final[index][1];
        }
    }

    time += 0.15;

    render();
}

function render(){
    webglUtils.resizeCanvasToDisplaySize(gl.canvas);

    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Clear the canvas.
    gl.clearColor(255, 255, 255, 255);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);

    wave_height_field_final_flat = wave_height_field_final.flatMap((vec) => ([vec[0], vec[1], vec[2]]));
    wave_normal_field_final_flat = wave_normal_field_final.flatMap((vec) => ([vec[0], vec[1], vec[2]]));

    updateViewMatrix();

    // Set uniforms
    gl.uniformMatrix4fv(viewMatrixLocation, false, viewMatrix);
    gl.uniformMatrix4fv(projectionMatrixLocation, false, projectionMatrix);
    gl.uniform3fv(cameraPosition, camera.eye);
    gl.uniform3fv(lightPosition, [0, 20, 0]);

    // Wave heights
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(wave_height_field_final_flat), gl.STATIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    gl.vertexAttribPointer(positionsLocation, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(positionsLocation);

    // Wave normals 
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(wave_normal_field_final_flat), gl.STATIC_DRAW);

    gl.vertexAttribPointer(normalsLocation, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(normalsLocation);

    // Draw
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);
    /*
    var primitiveType = gl.POINTS;
    var offset = 0;
    var count = wave_height_field_final_flat.length / 3; // 3 components (x, y, z) per vertex 
    gl.drawArrays(primitiveType, offset, count);
    */

    requestAnimationFrame(simulateOcean);
}




document.addEventListener("DOMContentLoaded", function() {

    const startButton = document.getElementById("btnStartSimulation");
    startButton.addEventListener('click', function(event){
        main();
    });

 });