import GUI from './Scripts/muigui.module.js';

function createFVertices() {



    /*
      pixel space => +Y is down
      clip space => +Y is up
    */


    const vertexData = new Float32Array([
      // left column
      0, 0,
      30, 0,
      0, 150,
      30, 150,
  
      // top rung
      30, 0,
      100, 0,
      30, 30,
      100, 30,
  
      // middle rung
      30, 60,
      70, 60,
      30, 90,
      70, 90,
    ]);
  
    const indexData = new Uint32Array([
      0,  1,  2,    2,  1,  3,  // left column
      4,  5,  6,    6,  5,  7,  // top run
      8,  9, 10,   10,  9, 11,  // middle run
    ]);
  
    return {
      vertexData,
      indexData,
      numVertices: indexData.length,
    };
  }

  const mat3 = {
    projection(width, height, dst) {
      // Note: This matrix flips the Y axis so that 0 is at the top.
      dst = dst || new Float32Array(12);
      dst[0] = 2 / width;  dst[1] = 0;             dst[2] = 0;
      dst[4] = 0;          dst[5] = -2 / height;   dst[6] = 0;
      dst[8] = -1;         dst[9] = 1;             dst[10] = 1;
      return dst;
    },
  
    identity(dst) {
      dst = dst || new Float32Array(12);
      dst[0] = 1;  dst[1] = 0;  dst[2] = 0;
      dst[4] = 0;  dst[5] = 1;  dst[6] = 0;
      dst[8] = 0;  dst[9] = 0;  dst[10] = 1;
      return dst;
    },
  
    multiply(a, b, dst) {
      dst = dst || new Float32Array(12);
      const a00 = a[0 * 4 + 0];
      const a01 = a[0 * 4 + 1];
      const a02 = a[0 * 4 + 2];
      const a10 = a[1 * 4 + 0];
      const a11 = a[1 * 4 + 1];
      const a12 = a[1 * 4 + 2];
      const a20 = a[2 * 4 + 0];
      const a21 = a[2 * 4 + 1];
      const a22 = a[2 * 4 + 2];
      const b00 = b[0 * 4 + 0];
      const b01 = b[0 * 4 + 1];
      const b02 = b[0 * 4 + 2];
      const b10 = b[1 * 4 + 0];
      const b11 = b[1 * 4 + 1];
      const b12 = b[1 * 4 + 2];
      const b20 = b[2 * 4 + 0];
      const b21 = b[2 * 4 + 1];
      const b22 = b[2 * 4 + 2];
  
      dst[ 0] = b00 * a00 + b01 * a10 + b02 * a20;
      dst[ 1] = b00 * a01 + b01 * a11 + b02 * a21;
      dst[ 2] = b00 * a02 + b01 * a12 + b02 * a22;
  
      dst[ 4] = b10 * a00 + b11 * a10 + b12 * a20;
      dst[ 5] = b10 * a01 + b11 * a11 + b12 * a21;
      dst[ 6] = b10 * a02 + b11 * a12 + b12 * a22;
  
      dst[ 8] = b20 * a00 + b21 * a10 + b22 * a20;
      dst[ 9] = b20 * a01 + b21 * a11 + b22 * a21;
      dst[10] = b20 * a02 + b21 * a12 + b22 * a22;
      return dst;
    },
  
    translation([tx, ty], dst) {
      dst = dst || new Float32Array(12);
      dst[0] = 1;   dst[1] = 0;   dst[2] = 0;
      dst[4] = 0;   dst[5] = 1;   dst[6] = 0;
      dst[8] = tx;  dst[9] = ty;  dst[10] = 1;
      return dst;
    },
  
    rotation(angleInRadians, dst) {
      const c = Math.cos(angleInRadians);
      const s = Math.sin(angleInRadians);
      dst = dst || new Float32Array(12);
      dst[0] = c;   dst[1] = s;  dst[2] = 0;
      dst[4] = -s;  dst[5] = c;  dst[6] = 0;
      dst[8] = 0;   dst[9] = 0;  dst[10] = 1;
      return dst;
  
    },
  
    scaling([sx, sy], dst) {
      dst = dst || new Float32Array(12);
      dst[0] = sx;  dst[1] = 0;   dst[2] = 0;
      dst[4] = 0;   dst[5] = sy;  dst[6] = 0;
      dst[8] = 0;   dst[9] = 0;   dst[10] = 1;
      return dst;
    },
  
    translate(m, translation, dst) {
      return mat3.multiply(m, mat3.translation(translation), dst);
    },
  
    rotate(m, angleInRadians, dst) {
      return mat3.multiply(m, mat3.rotation(angleInRadians), dst);
    },
  
    scale(m, scale, dst) {
      return mat3.multiply(m, mat3.scaling(scale), dst);
    },
  };
  
  
  async function main() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
      fail('need a browser that supports WebGPU');
      return;
    }
  
    // Get a WebGPU context from the canvas and configure it
    const canvas = document.querySelector('canvas');
    const context = canvas.getContext('webgpu');
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });
  
    const module = device.createShaderModule({
      code: `
        struct Uniforms {
          color: vec4f,
          matrix: mat3x3f,
        };
  
        struct Vertex {
          @location(0) position: vec2f,
        };
  
        struct VSOutput {
          @builtin(position) position: vec4f,
        };
  
        @group(0) @binding(0) var<uniform> uni: Uniforms;
  
        @vertex fn vs(vert: Vertex) -> VSOutput {
          var vsOut: VSOutput;
  
          let clipSpace = (uni.matrix * vec3f(vert.position, 1)).xy;
 
          vsOut.position = vec4f(clipSpace, 0.0, 1.0);
          return vsOut;
        }
  
        @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
          return uni.color;
        }
      `,
    });
  
    const pipeline = device.createRenderPipeline({
      label: 'just 2d position',
      layout: 'auto',
      vertex: {
        module,
        buffers: [
          {
            arrayStride: (2) * 4, // (2) floats, 4 bytes each
            attributes: [
              {shaderLocation: 0, offset: 0, format: 'float32x2'},  // position
            ],
          },
        ],
      },
      fragment: {
        module,
        targets: [{ format: presentationFormat }],
      },
      /*
      // dont draw triangles that are facing back (cull => dont draw)
      primitive: {
        cullMode: 'back',
      },
      */
      /*
      // depth testing - correctly draw based on Z value (if Z value of new pixel is less than of drawn one => draw, else dont)
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
      */
    });
  
    // color, matrix
    const uniformBufferSize = (4 + 12) * 4;
    const uniformBuffer = device.createBuffer({
      label: 'uniforms',
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformValues = new Float32Array(uniformBufferSize / 4); // divide by 4 to convert bytes into floats (1 float = 4byte) 
  
    // offsets to the various uniform values in float32 indices
    const kColorOffset = 0;
    const kMatrixOffset = 4;
  
    const colorValue = uniformValues.subarray(kColorOffset, kColorOffset + 4);
    const matrixValue = uniformValues.subarray(kMatrixOffset, kMatrixOffset + 12);
  
    // The color will not change so let's set it once at init time
    colorValue.set([Math.random(), Math.random(), Math.random(), 1]);

    const bindGroup = device.createBindGroup({
      label: 'bind group for object',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer }},
      ],
    });
  
    const { vertexData, indexData, numVertices } = createFVertices();
    const vertexBuffer = device.createBuffer({
      label: 'vertex buffer vertices',
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertexData);
    const indexBuffer = device.createBuffer({
      label: 'index buffer',
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indexData);
  
    const renderPassDescriptor = {
      label: 'our basic canvas renderPass',
      colorAttachments: [
        {
          // view: <- to be filled out when we render
          //clearValue: [0.3, 0.3, 0.3, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      /*
      // For depth testing
      depthStencilAttachment: {
        // view: <- to be filled out when we render
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      */
    };

    const degToRad = d => d * Math.PI / 180;
  
    const settings = {
      translation: [150, 100],
      rotation: degToRad(30),
      scale: [1, 1],
    };
  
    const radToDegOptions = { min: -360, max: 360, step: 1, converters: GUI.converters.radToDeg };

    const gui = new GUI();
    gui.onChange(render);
    gui.add(settings.translation, '0', 0, 1000).name('translation.x');
    gui.add(settings.translation, '1', 0, 1000).name('translation.y');
    gui.add(settings, 'rotation', radToDegOptions);
    gui.add(settings.scale, '0', -5, 5).name('scale.x');
    gui.add(settings.scale, '1', -5, 5).name('scale.y');
  
    function render() {
      // Get the current texture from the canvas context and
      // set it as the texture to render to.
      renderPassDescriptor.colorAttachments[0].view =
          context.getCurrentTexture().createView();
  
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass(renderPassDescriptor);
      pass.setPipeline(pipeline);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, 'uint32');

      mat3.projection(canvas.clientWidth, canvas.clientHeight, matrixValue);
      mat3.translate(matrixValue, settings.translation, matrixValue);
      mat3.rotate(matrixValue, settings.rotation, matrixValue);
      mat3.scale(matrixValue, settings.scale, matrixValue);
  
      // upload the uniform values to the uniform buffer
      device.queue.writeBuffer(uniformBuffer, 0, uniformValues);
  
      pass.setBindGroup(0, bindGroup);
      pass.drawIndexed(numVertices);
  
      pass.end();
  
      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
    }
  
    const observer = new ResizeObserver(entries => {
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
  
  function fail(msg) {
    alert(msg);
  }
  
  document.addEventListener("DOMContentLoaded", function() {
  
      main();
  
   });