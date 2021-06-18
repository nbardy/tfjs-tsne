import * as tfwebgl from "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

export const getWebGLBackend = () => {
  const backendType = tf.getBackend();
  const backend = tf.backend();
  if (backendType !== "webgl") {
    throw Error("WebGB backend is not available");
  }

  return backend as tfwebgl.MathBackendWebGL;
};
