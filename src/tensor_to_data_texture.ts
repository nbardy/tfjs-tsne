/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from "@tensorflow/tfjs-core";
// import * as gpgpu from "@tensorflow/tfjs-backend-webgl/src/gpgpu_util";
import { RearrangedData } from "./interfaces";

// import * as tsne from "./tsne";

import * as webgl_util from "@tensorflow/tfjs-backend-webgl/src/webgl_util";
import { getWebGLBackend } from "./getWebGLBackend";

// I copied this from tfjs - Art Brain
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

// This functions are not exported by DeepLearnJS but are need from the
// functions that I am hacking to access the backend.
// Ideally my functions will be part of DeepLearnJS and there will be no need
// to have identical external copies

///////////////////////  GPGPU_UTIL  //////////////////////////////
// EXACT COPY DLJS
function getTextureInternalFormat(
  gl: WebGLRenderingContext,
  numChannels: number
): number {
  if (numChannels === 4) {
    // tslint:disable-next-line:no-any
    return (gl as any).RGBA32F;
  } else if (numChannels === 3) {
    // tslint:disable-next-line:no-any
    return (gl as any).RGB32F;
  } else if (numChannels === 2) {
    // tslint:disable-next-line:no-any
    return (gl as any).RG32F;
  }
  // tslint:disable-next-line:no-any
  return (gl as any).R32F;
}

// function getTextureInternalUByteFormat(
//   gl: WebGLRenderingContext,
//   numChannels: number
// ): number {
//   if (numChannels === 4) {
//     // tslint:disable-next-line:no-any
//     return (gl as any).RGBA8;
//   } else if (numChannels === 3) {
//     // tslint:disable-next-line:no-any
//     return (gl as any).RGB8;
//   } else if (numChannels === 2) {
//     // tslint:disable-next-line:no-any
//     return (gl as any).RG8;
//   }
//   // tslint:disable-next-line:no-any
//   return (gl as any).R8;
// }

function getTextureFormat(
  gl: WebGLRenderingContext,
  numChannels: number
): number {
  if (numChannels === 4) {
    // tslint:disable-next-line:no-any
    return (gl as any).RGBA;
  } else if (numChannels === 3) {
    // tslint:disable-next-line:no-any
    return (gl as any).RGB;
  } else if (numChannels === 2) {
    // tslint:disable-next-line:no-any
    return (gl as any).RG;
  }
  // tslint:disable-next-line:no-any
  return (gl as any).RED;
}
// EXACT COPY DLJS
function getTextureType(gl: WebGLRenderingContext) {
  return gl.FLOAT;
}

// function getTextureTypeUByte(gl: WebGLRenderingContext) {
//   return gl.UNSIGNED_BYTE;
// }

function createAndConfigureTexture(
  gl: WebGLRenderingContext,
  width: number,
  height: number,
  numChannels: number,
  pixels: ArrayBufferView
): WebGLTexture {
  webgl_util.validateTextureSize(width, height);
  const texture = webgl_util.createTexture(gl);

  const tex2d = gl.TEXTURE_2D;
  const internalFormat = getTextureInternalFormat(gl, numChannels);
  const format = getTextureFormat(gl, numChannels);
  const textureType = getTextureType(gl);

  getTextureFormat;
  webgl_util.callAndCheck(gl, () => gl.bindTexture(tex2d, texture));
  webgl_util.callAndCheck(gl, () =>
    gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  );
  webgl_util.callAndCheck(gl, () =>
    gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  );
  webgl_util.callAndCheck(gl, () =>
    gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  );
  webgl_util.callAndCheck(gl, () =>
    gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  );
  webgl_util.callAndCheck(gl, () =>
    gl.texImage2D(
      tex2d,
      0,
      internalFormat,
      width,
      height,
      0,
      format,
      textureType,
      pixels
    )
  );
  webgl_util.callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
  return texture;
}

/*******
 * Converts a 2D tensor in a texture that is in the optimized format
 * for the kNN computation
 * @param {tfc.Tensor} Tensor to convert
 * @return {Promise} Promise of an Obeject containing the texture and shape
 */
export async function tensorToDataTexture(
  tensor: tf.Tensor
): Promise<{ shape: RearrangedData; texture: WebGLTexture }> {
  const inputShape = tensor.shape;
  if (inputShape.length !== 2) {
    throw Error("tensorToDataTexture: input tensor must be 2-dimensional");
  }

  const backend = getWebGLBackend();

  const gpgpu = backend.getGPGPUContext();
  const gl = gpgpu.gl;
  console.log("GPGPU", gpgpu);

  // Computing texture shape
  const numPoints = inputShape[0];
  const numDimensions = inputShape[1];
  const numChannels = 4;
  const pixelsPerPoint = Math.ceil(numDimensions / numChannels);
  const pointsPerRow = Math.max(
    1,
    Math.floor(Math.sqrt(numPoints * pixelsPerPoint) / pixelsPerPoint)
  );
  const numRows = Math.ceil(numPoints / pointsPerRow);

  const tensorData = tensor.dataSync();

  // TODO Switch to a GPU implmentation to improve performance
  // Reading tensor values
  const textureValues = new Float32Array(
    pointsPerRow * pixelsPerPoint * numRows * numChannels
  );

  for (let p = 0; p < numPoints; ++p) {
    const tensorOffset = p * numDimensions;
    const textureOffset = p * pixelsPerPoint * numChannels;
    for (let d = 0; d < numDimensions; ++d) {
      textureValues[textureOffset + d] = tensorData[tensorOffset + d];
    }
  }

  const texture = createAndConfigureTexture(
    gl,
    pointsPerRow * pixelsPerPoint,
    numRows,
    4,
    textureValues
  );
  const shape = { numPoints, pointsPerRow, numRows, pixelsPerPoint };

  return { shape, texture };
}
