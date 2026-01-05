const identityMatrix = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
];

const imageTransforms = [];

function setup() {
  // Use WEBGL so texture()/vertex(u,v) in drawImageWithHomography works
  canvas = createCanvas(1600, 800, WEBGL);
  canvas.drop(onFileDropped);

  createForegroundSegmenter();
  
  // ensure texture UVs use normalized coordinates
  textureMode(NORMAL);
}

var maskSegmentation = null;

function createForegroundSegmenter() {
  maskSegmentation = new SelfieSegmentation({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@latest/${file}`;
  }});
  var options = {
      selfieMode: true,
      modelSelection: 0,  //general
      effect: 'mask',
  };
  
  maskSegmentation.setOptions(options);
}

function onFileDropped(file) {
  const id = file.name;
  console.log("Dropped file: " + file.name);
  const div = upsertMedia(id);

  const originalImg = createImg(file.data, '', () => {
    originalImg.parent(div);
    originalImg.addClass('original');

    const lowResImg = generateLowResImage(originalImg.elt, () => {
      lowResImg.parent(div);
      lowResImg.addClass('lowres');

      const maskImg = generateMask(lowResImg.elt, () => {
        maskImg.parent(div);
        maskImg.addClass('mask');

        const foregroundImg = applyMaskToImage(lowResImg.elt, maskImg.elt, false, () => {
          foregroundImg.parent(div);
          foregroundImg.addClass('foreground');

          const backgroundImg = applyMaskToImage(lowResImg.elt, maskImg.elt, true, () => {
            backgroundImg.parent(div);
            backgroundImg.addClass('background');

            processMedia();
          });
        });
      });
    });
  });
}

function generateLowResImage(imgElement, onloaded = () => {}) {
  let lowresImg = null;

  const lowresMaxPixels = 640 * 480;
  if (imgElement.width * imgElement.height > lowresMaxPixels) { 
    const s = Math.sqrt(lowresMaxPixels / (imgElement.width * imgElement.height));

    const targetW = Math.round(imgElement.width * s);
    const targetH = Math.round(imgElement.height * s);
    
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, targetW, targetH);

    const dataUrl = canvas.toDataURL("image/jpeg", 1.0);
    
    // clear canvas to free memory immediately
    canvas.width = 0;
    canvas.height = 0;

    lowresImg = createImg(dataUrl, '', onloaded);
  }
  else {
    lowresImg = new p5.Element(imgElement);
    setTimeout(onloaded, 0);
  }

  return lowresImg;
}

function generateMask(imgElement, onloaded = () => {}) {
  let maskImg = createImg('', '');

  maskSegmentation.onResults(async (results) => {
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = results.segmentationMask.width;
      maskCanvas.height = results.segmentationMask.height;
      const ctx = maskCanvas.getContext('2d');
      
      // flip horizontally
      ctx.translate(maskCanvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(results.segmentationMask, 0, 0);

      // convert red-channel mask to greyscale (copy R to G and B)
      const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];     // red channel holds the mask value
        data[i]     = r;       // R (keep)
        data[i + 1] = r;       // G (copy from R)
        data[i + 2] = r;       // B (copy from R)
        //data[i + 3] = 255;     // A (fully opaque)
      }
      ctx.putImageData(imageData, 0, 0);

      maskImg.elt.onload = onloaded;
      maskImg.elt.src = maskCanvas.toDataURL();
    });
    maskSegmentation.send({ image: imgElement });

    return(maskImg);
}

function processMedia() {
  const mediaElement = select('#media')?.elt;
  if(mediaElement) {
    const newElementIndex = mediaElement.childElementCount - 1;
    const newElement = mediaElement.children[newElementIndex];

    if(newElementIndex > 0){
      let foundValidHomography = false;
      
      for(let n = newElementIndex - 1; n >= 0; n--) {
        console.log("Testing alignment between:" + newElementIndex + " and:" + n);
        Align_img(mediaElement.children[n].querySelector('.background'), newElement.querySelector('.background'));
        if(h && !h.empty() && h.data64F) {
          const check = isReasonableHomography(Array.from(h.data64F));
          console.log('Homography check:', check);
          
          if (check.valid) {
            // homography from new image to matching image n
            const localTransform = [
              h.data64F[0], h.data64F[1], 0, h.data64F[2],
              h.data64F[3], h.data64F[4], 0, h.data64F[5],
              0, 0, 1, 0,
              h.data64F[6], h.data64F[7], 0, h.data64F[8]
            ];
            
            // combine with the transform of the matching element
            const matchingTransform = imageTransforms[n] || identityMatrix;
            const combinedTransform = multiplyMatrix4x4(localTransform, matchingTransform);
            
            imageTransforms.push(combinedTransform);
            console.log("Computed combined transform from:" + newElementIndex + " to:" + n, combinedTransform);
            foundValidHomography = true;
            break;
          } else {
            console.warn('Rejecting homography:', check.reason);
            // continue loop to try next
          }
        }
      }
      
      // no valid homography found with any existing element
      if (!foundValidHomography) {
        imageTransforms.push(null);
        console.log("No valid homography found, setting transform to null for index:", newElementIndex);
      }
    }
    else {
      imageTransforms.push(identityMatrix);
    }
  }
}

// cache for converted images (HTMLImageElement -> p5.Graphics)
const textureCache = new WeakMap();

function getTextureFromElement(el) {
  if (!el) return null;
  
  // check cache first
  if (textureCache.has(el)) {
    const cached = textureCache.get(el);
    // check if image size changed (unlikely but safe)
    if (cached.width === el.width && cached.height === el.height) {
      return cached;
    }
    // size changed, remove old and recreate
    cached.remove();
  }
  
  // convert HTMLImageElement to p5.Graphics
  const g = createGraphics(el.width, el.height);
  g.drawingContext.drawImage(el, 0, 0);
  textureCache.set(el, g);
  return g;
}

// draw a textured quad: srcImg projected by homography Hproj into target image space (targetIndex)
function drawProjectedImage(srcImg, x, y, Hproj, zDepth = 0) {
  if (!srcImg || !Hproj) return;
  
  const img = getTextureFromElement(srcImg);
  if (!img) return;
  
  const w = img.width, h = img.height;
  const corners = [0,0,w,0,w,h,0,h];
  // project corners into target image pixel coords (corners is a flat array [x0,y0,...])
  const dst = [];
  for (let i = 0; i < corners.length; i += 2) {
    const p = applyTransform4x4(corners[i], corners[i + 1], Hproj) || [0, 0];
    dst.push(p[0]+x, p[1]+y);
  }
  // draw textured polygon in WEBGL using normalized texture coords (0..1)
  push();
    noStroke();
    texture(img);
    beginShape();
      // vertex(x, y, z, u, v)
      vertex(dst[0], dst[1], zDepth, 0, 0);
      vertex(dst[2], dst[3], zDepth, 1, 0);
      vertex(dst[4], dst[5], zDepth, 1, 1);
      vertex(dst[6], dst[7], zDepth, 0, 1);
    endShape(CLOSE);
  pop();
}

function upsertMedia(id) {
  if (!id) return null;

  let container = select('#media');
  if (!container) return null;

  const found = container.elt.querySelector('#' + id);
  if (found) return select('#' + id); // use p5.select to return a p5.Element

  const d = createDiv('');
  d.id(id);
  d.parent(container);
  return d;
}

/**
 * Finds the index of the image whose transformed origin (0,0) is closest to the mouse.
 * @returns {number} - index of closest image, or -1 if none
 */
function getClosestImageToMouse() {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement) return -1;

  // mouse position in WEBGL coordinates (origin at center of canvas)
  const mx = mouseX - width / 2;
  const my = mouseY - height / 2;

  let closestIndex = -1;
  let closestDist = Infinity;

  for (let i = 0; i < mediaElement.children.length; i++) {
    const transform = imageTransforms[i];
    if (!transform) continue;

    // transform origin (0,0) to screen coords
    const [tx, ty] = applyTransform4x4(0, 0, transform);
    const screenX = tx * 0.6;
    const screenY = ty * 0.6;

    const dist = Math.sqrt((mx - screenX) ** 2 + (my - screenY) ** 2);

    if (dist < closestDist) {
      closestDist = dist;
      closestIndex = i;
    }
  }

  return closestIndex;
}

function draw() {
  background(220);

  const closestImageIndex = getClosestImageToMouse();

  const mediaElement = select('#media')?.elt;
  if(mediaElement) {
    push();
      scale(0.6);
      //for each mediaElement.children[0].querySelector('.lowres')
      //console.log(' mediaElement:',  mediaElement);
      for (let i = 0; i < mediaElement.children.length; i++) {
        const image = mediaElement.children[i].querySelector('.lowres');
        if (image) {
          if(i == 0) translate(-image.width / 2, -image.height / 2);
          push();
            if(i === closestImageIndex) tint(255, 255);
            else tint(255, 127);
            // closest image drawn in front (lower z = closer to camera in default WEBGL)
            const zDepth = (i === closestImageIndex) ? 0 : -1;
            drawProjectedImage(image, 0, 0, imageTransforms[i], zDepth);
          pop();
        }
      }

      // for (let i = 0; i < mediaElement.children.length; i++) {
      //   const image = mediaElement.children[i].querySelector('.foreground');
      //   if (image) {
      //     //tint(255, 127);
      //     drawProjectedImage(image, 0, 0,  imageTransforms[i]);
      //   }
      // }

      // if(inputImageA) {
      //   push();
      //     tint(255, 127);
      //     drawProjectedImage(inputImageA, 0, 0,  imageTransforms[1]);
      //   pop();
      // }

      // if(inputImageB) {
      //   push();
      //     tint(255, 127);
      //     drawProjectedImage(inputImageB, 0, 0, imageTransforms[0]);
      //   pop();
      // }

      /*
      push();
        // draw matches overlay last so lines appear on top
        // keep this inside the same top-left transform so coordinates match the points/circles
        try {
          const gl = drawingContext;
          if (gl && gl.disable) gl.disable(gl.DEPTH_TEST);
        } catch (e) { }
        drawMatchesOverlay();
        try {
          const gl = drawingContext;
          if (gl && gl.enable) gl.enable(gl.DEPTH_TEST);
        } catch (e) {}
      // close the initial top-left transform
      pop();
      */
    pop();
  }
}

function applyTransform4x4(px, py, M) {
  // strict: accept only flat row-major 4x4 arrays (length 16)
  if (!Array.isArray(M) || M.length !== 16) return [px, py];

  const X = M[0] * px + M[1] * py + M[2] * 0 + M[3];
  const Y = M[4] * px + M[5] * py + M[6] * 0 + M[7];
  const W = M[12] * px + M[13] * py + M[14] * 0 + M[15];

  if (!isFinite(W) || Math.abs(W) < 1e-12) return [X, Y];
  return [X / W, Y / W];
}

/**
 * Creates a new image element with the mask applied.
 * Pixels where the mask is dark (black) become transparent.
 * @param {HTMLImageElement|p5.Element} colorImg - the colour image
 * @param {HTMLImageElement|p5.Element} maskImg - the greyscale mask (white = keep, black = transparent)
 * @returns {p5.Element} - a new p5 img element containing the masked image
 */
function applyMaskToImage(colorImg, maskImg, invert = false, onloaded = () => {}) {
  let resultImg = createImg('', '');

  const w = colorImg.naturalWidth || colorImg.width;
  const h = colorImg.naturalHeight || colorImg.height;

  // create a canvas to composite the result
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  // draw the colour image first
  ctx.drawImage(colorImg, 0, 0, w, h);

  // get colour image data
  const colorData = ctx.getImageData(0, 0, w, h);
  const cPixels = colorData.data;

  // draw the mask (scaled to same size)
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(maskImg, 0, 0, w, h);
  const maskData = ctx.getImageData(0, 0, w, h);
  const mPixels = maskData.data;

  // apply mask: use mask's red channel as alpha
  for (let i = 0; i < cPixels.length; i += 4) {
    // mask value (0 = transparent, 255 = opaque)
    const maskVal = invert ? 255 - mPixels[i] : mPixels[i]; // red channel of mask

    cPixels[i] = maskVal > 0 ? cPixels[i] : random(255);
    cPixels[i + 1] = maskVal > 0 ? cPixels[i + 1] : random(255);
    cPixels[i + 2] = maskVal > 0 ? cPixels[i + 2] : random(255);
    cPixels[i + 3] = maskVal;   // set alpha of colour pixel
  }

  // put the masked result back
  ctx.putImageData(colorData, 0, 0);

  // create a new p5 image element from the canvas
  resultImg.elt.onload = onloaded;
  resultImg.elt.src = canvas.toDataURL();

  return resultImg;
}

/**
 * Checks if a homography transform looks reasonable.
 * Returns { valid: boolean, reason: string, rotation: number, scale: number, shear: number }
 * 
 * A "reasonable" homography for image alignment should have:
 * - Minimal rotation (< maxRotationDeg)
 * - Scale close to 1 (within scaleRange)
 * - Low shear
 * - Low perspective distortion (bottom row close to [0, 0, 1])
 * 
 * @param {Array} H - flat 9-element row-major 3x3 homography, or flat 16-element 4x4
 * @param {Object} options - optional thresholds
 * @returns {Object} { valid, reason, rotation, scale, shear, perspective }
 */
function isReasonableHomography(H, options = {}) {
  const {
    maxRotationDeg = 15,      // max allowed rotation in degrees
    minScale = 0.5,           // min allowed scale
    maxScale = 2.0,           // max allowed scale
    maxShear = 0.3,           // max allowed shear
    maxPerspective = 0.001    // max allowed perspective distortion
  } = options;

  if (!H) return { valid: false, reason: 'H is null or undefined' };

  // extract 3x3 from flat 9 or flat 16
  let h00, h01, h02, h10, h11, h12, h20, h21, h22;
  if (H.length === 9) {
    [h00, h01, h02, h10, h11, h12, h20, h21, h22] = H;
  } else if (H.length === 16) {
    // 4x4 row-major: extract the 2D affine/projective part
    h00 = H[0];  h01 = H[1];  h02 = H[3];   // skip H[2] (z column)
    h10 = H[4];  h11 = H[5];  h12 = H[7];
    h20 = H[12]; h21 = H[13]; h22 = H[15];
  } else {
    return { valid: false, reason: 'H must be length 9 or 16' };
  }

  // normalize so h22 = 1 (if possible)
  if (Math.abs(h22) < 1e-12) {
    return { valid: false, reason: 'h22 is zero, degenerate homography' };
  }
  h00 /= h22; h01 /= h22; h02 /= h22;
  h10 /= h22; h11 /= h22; h12 /= h22;
  h20 /= h22; h21 /= h22; h22 = 1;

  // perspective distortion: bottom row should be [0, 0, 1]
  const perspective = Math.sqrt(h20 * h20 + h21 * h21);
  if (perspective > maxPerspective) {
    return {
      valid: false,
      reason: `Perspective distortion too high: ${perspective.toFixed(6)} > ${maxPerspective}`,
      perspective
    };
  }

  // decompose upper-left 2x2 into rotation, scale, shear
  // H = [ a  b  tx ]   where [a b; c d] = R * S * Shear
  //     [ c  d  ty ]
  //     [ 0  0  1  ]
  const a = h00, b = h01, c = h10, d = h11;

  // scale: sqrt of determinant gives overall scale
  const det = a * d - b * c;
  if (det <= 0) {
    return { valid: false, reason: 'Negative or zero determinant (flipped or degenerate)' };
  }
  const scale = Math.sqrt(det);

  // rotation angle from the 2x2 matrix (assumes no/low shear)
  // rotation = atan2(c, a) for a proper rotation matrix
  const rotationRad = Math.atan2(c, a);
  const rotationDeg = Math.abs(rotationRad * 180 / Math.PI);

  // shear: measure how non-orthogonal the axes are
  // shear ~ (a*b + c*d) / det for normalized matrix
  const shear = Math.abs(a * b + c * d) / det;

  // check thresholds
  if (rotationDeg > maxRotationDeg) {
    return {
      valid: false,
      reason: `Rotation too large: ${rotationDeg.toFixed(2)}° > ${maxRotationDeg}°`,
      rotation: rotationDeg,
      scale,
      shear,
      perspective
    };
  }

  if (scale < minScale || scale > maxScale) {
    return {
      valid: false,
      reason: `Scale out of range: ${scale.toFixed(3)} not in [${minScale}, ${maxScale}]`,
      rotation: rotationDeg,
      scale,
      shear,
      perspective
    };
  }

  if (shear > maxShear) {
    return {
      valid: false,
      reason: `Shear too high: ${shear.toFixed(3)} > ${maxShear}`,
      rotation: rotationDeg,
      scale,
      shear,
      perspective
    };
  }

  return {
    valid: true,
    reason: 'OK',
    rotation: rotationDeg,
    scale,
    shear,
    perspective
  };
}

/**
 * Multiplies two 4x4 row-major flat matrices and returns the result.
 * Result = A * B (A applied first, then B)
 * @param {Array} A - flat 16-element row-major 4x4 matrix
 * @param {Array} B - flat 16-element row-major 4x4 matrix
 * @returns {Array} - flat 16-element row-major 4x4 matrix (A * B)
 */
function multiplyMatrix4x4(A, B) {
  if (!A || A.length !== 16 || !B || B.length !== 16) {
    console.warn('multiplyMatrix4x4: invalid input, returning identity');
    return [...identityMatrix];
  }

  const result = new Array(16);

  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += A[row * 4 + k] * B[k * 4 + col];
      }
      result[row * 4 + col] = sum;
    }
  }

  return result;
}

/**
 * Creates a 4x4 transform that places a new element to the right of an existing one.
 * @param {HTMLElement} existingEl - the element already placed (e.g. mediaElement.children[n])
 * @param {HTMLElement} newEl - the new element to place to the right
 * @param {Array} existingTransform - the 4x4 transform of the existing element (flat 16-element array)
 * @param {string} imgClass - CSS class of the image to measure (e.g. 'lowres', 'foreground', 'background')
 * @param {number} gap - optional gap in pixels between elements (default 10)
 * @returns {Array} - flat 16-element 4x4 row-major transform matrix
 */
function getTileRightTransform(existingEl, newEl, existingTransform) {  
  if (!existingEl || !newEl) {
    console.warn('getTileRightTransform: missing .' + imgClass + ' elements');
    return [...identityMatrix];
  }

  const existingWidth = existingEl.naturalWidth || existingEl.width || 0;
  
  // extract translation from existing transform (if any)
  let existingTx = 0, existingTy = 0;
  if (existingTransform && existingTransform.length === 16) {
    existingTx = existingTransform[3];
    existingTy = existingTransform[7];
  }

  // new element's X = existing X + existing width + gap
  const newTx = existingTx + existingWidth + gap;
  const newTy = existingTy; // same Y position

  // return identity with translation applied
  return [
    1, 0, 0, newTx,
    0, 1, 0, newTy,
    0, 0, 1, 0,
    0, 0, 0, 1
  ];
}