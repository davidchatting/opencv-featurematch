let bastoncini, bastoncini_in_scene;
let result;

function preload() {
  bastoncini = loadImage('images/bastoncini.png');
  bastoncini_in_scene = loadImage('images/bastoncini_in_scene.png');
}

async function setup() {
  createCanvas(800, 400, WEBGL);
  await featurematchReady();
  const options = {};
  result = alignImages(bastoncini.canvas, bastoncini_in_scene.canvas, options);
  console.log(result);
}

function draw() {
  background(220);
  translate(-width/2, -height/2);
  image(bastoncini, 0, 0);
  
  push();
    translate(bastoncini.width, 0);
    image(bastoncini_in_scene, 0, 0);
    if (result && result.valid) {
      applyMatrix(result.transform);
      noFill();
      strokeWeight(3);
      stroke('green');
      rect(0, 0, bastoncini.width, bastoncini.height);
    }
  pop();
}