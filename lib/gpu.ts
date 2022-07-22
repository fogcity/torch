import co from "./shaders/compute.wgsl?raw";
(async () => {
  if (!("gpu" in navigator)) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.log("Failed to get GPU adapter.");
    return;
  }
  const device = await adapter.requestDevice();

  // First Matrix

  const firstMatrix = new Float32Array([
    2 /* rows */, 4 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8,
  ]);

  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });

  new Float32Array(gpuBufferFirstMatrix.getMappedRange()).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix

  const secondMatrix = new Float32Array([
    4 /* rows */, 2 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8,
  ]);

  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });

  new Float32Array(gpuBufferSecondMatrix.getMappedRange()).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix

  console.log("Float32Array.BYTES_PER_ELEMENT", Float32Array.BYTES_PER_ELEMENT);

  const gpuBufferResultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
  console.log(gpuBufferResultMatrixBufferSize);

  const gpuBufferResultMatrixBuffer = device.createBuffer({
    size: gpuBufferResultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Pipeline setup
  const computePipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: co,
      }),
      entryPoint: "main",
    },
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: gpuBufferResultMatrixBuffer,
        },
      },
    ],
  });

  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroupCountX = Math.ceil(firstMatrix[0] / 8);
  const workgroupCountY = Math.ceil(secondMatrix[1] / 8);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: gpuBufferResultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    gpuBufferResultMatrixBuffer,
    0,
    gpuReadBuffer,
    0,
    gpuBufferResultMatrixBufferSize
  );

  // Submit GPU commands.
  device.queue.submit([commandEncoder.finish()]);

  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  console.log(new Float32Array(arrayBuffer.slice(0)));
})();
