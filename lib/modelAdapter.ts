export type ModelPrediction = {
  label: string;
  confidence: number;
};

type Landmark = {
  x: number;
  y: number;
  z: number;
};

let loaded = false;
let model: unknown = null;

/*
 FUTURE:
 Replace this loader with TensorFlow.js model load:
 model = await tf.loadLayersModel("/models/isl/model.json")
*/

export async function loadGestureModel(): Promise<void> {
  if (loaded) return;

  try {
    model = { ready: true };
    loaded = true;
  } catch (error) {
    console.error("Model load failed", error);
    loaded = false;
  }
}

export function isGestureModelReady(): boolean {
  return loaded && !!model;
}

/*
 Converts 21 landmarks into flat numeric vector
 [x1,y1,z1,x2,y2,z2 ...]
*/
export function landmarksToVector(hand: Landmark[]): number[] {
  const vector: number[] = [];

  for (const point of hand) {
    vector.push(point.x, point.y, point.z);
  }

  return vector;
}

/*
 Temporary inference bridge.
 Replace with tfjs prediction later.
*/
export async function predictGesture(
  hand: Landmark[]
): Promise<ModelPrediction | null> {
  if (!isGestureModelReady()) return null;

  const vector = landmarksToVector(hand);

  if (vector.length < 63) return null;

  /*
    PLACEHOLDER OUTPUTS
    Keeps pipeline alive safely
  */

  const openPalm = hand[8].y < hand[6].y &&
                   hand[12].y < hand[10].y &&
                   hand[16].y < hand[14].y &&
                   hand[20].y < hand[18].y;

  const fist = hand[8].y > hand[6].y &&
               hand[12].y > hand[10].y &&
               hand[16].y > hand[14].y &&
               hand[20].y > hand[18].y;

  if (openPalm) {
    return {
      label: "Hello",
      confidence: 0.95,
    };
  }

  if (fist) {
    return {
      label: "Stop",
      confidence: 0.93,
    };
  }

  return {
    label: "Gesture",
    confidence: 0.82,
  };
}