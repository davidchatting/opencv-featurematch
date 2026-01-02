const imageTransforms = [
  [1, 0, 0, 0,
   0, 1, 0, 0,
   0, 0, 1, 0,
   0, 0, 0, 1],
  [1, 0, 0, 100,
   0, 1, 0,   0,
   0, 0, 1,   0,
   0, 0, 0,   1]
];

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
  console.log("Dropped file: " + file.name);
  const div = upsertMedia(file.name);

  const img = createImg(file.data, '', () => {
    img.parent(div);
    img.addClass('original');

    const lowResImg = generateLowResImage(img.elt, () => {
      lowResImg.parent(div);
      lowResImg.addClass('lowres');

      const maskImg = generateMask(lowResImg.elt, () => {
        maskImg.parent(div);
        maskImg.addClass('mask');

        const foregroundImg = applyMaskToImage(lowResImg, maskImg, false, () => {
          foregroundImg.parent(div);
          foregroundImg.addClass('foreground');

          const backgroundImg = applyMaskToImage(lowResImg, maskImg, true, () => {
            backgroundImg.parent(div);
            backgroundImg.addClass('background');
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

function processImages() {
  const mediaElement = select('#media')?.elt;
  if(mediaElement) {
    const n = mediaElement.childElementCount;
    
    if(n === 2){
      console.log("Two images loaded, starting alignment...");
      Align_img(mediaElement.children[0].querySelector('.lowres') , mediaElement.children[1].querySelector('.lowres'));
      if(h && !h.empty() && h.data64F) {
        const d = h.data64F;
        //flat 4x4 row-major so
        imageTransforms[1] = [
          d[0], d[1], 0, d[2],
          d[3], d[4], 0, d[5],
          0   , 0   , 1, 0   ,
          d[6], d[7], 0, d[8]
        ];
      }
    }
  }
}

  // draw a textured quad: srcImg projected by homography Hproj into target image space (targetIndex)
  function drawProjectedImage(srcImg, x, y, Hproj) {
    if (!srcImg || !Hproj) return;
    const w = srcImg.width, h = srcImg.height;
    const corners = [0,0,w,0,w,h,0,h];
    // project corners into target image pixel coords (corners is a flat array [x0,y0,...])
    const dst = [];
    for (let i = 0; i < corners.length; i += 2) {
      const p = applyTransform4x4(corners[i], corners[i + 1], Hproj) || [0, 0];
      dst.push(p[0]+x, p[1]+y);
    }
    // draw textured polygon in WEBGL using normalized texture coords (0..1)
    push();
      try {
        const gl = drawingContext;
        if (gl && gl.disable) gl.disable(gl.DEPTH_TEST);
      } catch (e) {}
      noStroke();
      texture(srcImg);
      beginShape();
        // top-left (0,0) -> u=0,v=0 ; top-right -> u=1,v=0 ; bottom-right -> u=1,v=1 ; bottom-left -> u=0,v=1
        vertex(dst[0], dst[1], 0, 0);
        vertex(dst[2], dst[3], 1, 0);
        vertex(dst[4], dst[5], 1, 1);
        vertex(dst[6], dst[7], 0, 1);
      endShape(CLOSE);
      try {
        const gl = drawingContext;
        if (gl && gl.enable) gl.enable(gl.DEPTH_TEST);
      } catch (e) {}
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

function draw() {
  background(220);
  
  push();
    if(inputImageB) translate(-inputImageB.width / 2, -inputImageB.height / 2);
    if(inputImageA) {
      push();
        //tint(255, 127);
        drawProjectedImage(inputImageA, 0, 0,  imageTransforms[1]);
      pop();
    }

    if(inputImageB) {
      push();
        //tint(255, 127);
        drawProjectedImage(inputImageB, 0, 0, imageTransforms[0]);
      pop();
    }

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
  pop();
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

  // accept p5.Element or raw DOM element
  if (colorImg && colorImg.elt) colorImg = colorImg.elt;
  if (maskImg && maskImg.elt) maskImg = maskImg.elt;

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

    cPixels[i] = maskVal > 0 ? cPixels[i] : 0;
    cPixels[i + 1] = maskVal > 0 ? cPixels[i + 1] : 0;
    cPixels[i + 2] = maskVal > 0 ? cPixels[i + 2] : 0;
    cPixels[i + 3] = maskVal;   // set alpha of colour pixel
  }

  // put the masked result back
  ctx.putImageData(colorData, 0, 0);

  // create a new p5 image element from the canvas
  resultImg.elt.onload = onloaded;
  resultImg.elt.src = canvas.toDataURL();

  return resultImg;
}