var image_A_element = null;
var image_B_element = null;
var button;
var canvas;

var inputImageA = null, inputImageB = null;
var points1 = [];
var points2 = [];
var good_inlier_matches;
var h;

function preload() {
  image_A_element = document.getElementById('image_A_element_id');
  image_A_element.setAttribute('crossOrigin', 'Anonymous');
  image_A_element.setAttribute('src', 'images/box.png');
  //image_A_element.classList.add('hide');

  image_B_element = document.getElementById('image_B_element_id');
  image_B_element.setAttribute('crossOrigin', 'Anonymous');
  image_B_element.setAttribute('src', 'images/box_in_scene.png');
  //image_B_element.classList.add('hide');
  
  //must be a beter way to do this...
  //setTimeout(setupWhenImageLoaded,100);
}

function setup() {
  canvas = createCanvas(800, 800);
  canvas.parent("p5jsCanvas");
  
  button = createButton('click me');
  button.position(0, 0);
  button.mousePressed(buttonClick);
}

function setupWhenImageLoaded() {
  //Align_img();
}

function buttonClick() {
  Align_img();
}

function draw() {
  background(220);
  
  push();
  if(inputImageA) {
    tint(255, 127);
    push();
    translate(150, 50); 
    //translate(width/2, height/2); 
    //scale(min(width/inputImageA.width, height/inputImageA.height));  //scale the image to fit the canvas
    //translate(-inputImageA.width/2, -inputImageA.height/2);          //centre the image

    push();
    //TODO: why are these images reversed?
    scale(-1,1);
    translate(-inputImageA.width,0);
    image(inputImageA, 0, 0, inputImageA.width, inputImageA.height);
    pop();
    //console.log(inputImageA.width, inputImageA.height);
    
    for(var i=0;i<(points1.length/2);++i){
        circle(points1[(i*2)+0],points1[(i*2)+1],10);
    }
    
    pop();
  }
  
  if(inputImageB) {
    tint(255, 127);
    push();
    translate(150, 500); 
    //translate(width/2, height/2);
    
    //scale(min(width/inputImageB.width, height/inputImageB.height));  //scale the image to fit the canvas
    //translate(-inputImageB.width/2, -inputImageB.height/2);          //centre the image
    /*
    if(h && !h.empty()) {
      applyMatrix(
        h.data64F[0], h.data64F[1], h.data64F[2],
        h.data64F[3], h.data64F[4], h.data64F[5],
        h.data64F[6], h.data64F[7], h.data64F[8],
      );
    }
    */
    if(points2.length > 0) {
      //console.log(points2[0], points2[1])
    }
    for(var i=0;i<(points2.length/2);++i){
        circle(points2[(i*2)+0],points2[(i*2)+1],10);
    }
    
    if(h && !h.empty()) {
      drawImageWithHomography(inputImageB, h.data64F);
    }
    
    push();
    //TODO: why are these images reversed?
    scale(-1,1);
    translate(-inputImageB.width,0);
    image(inputImageB, 0, 0, inputImageB.width, inputImageB.height);
    pop();
    
    pop();
  }
  pop();
}

function Align_img() {
  //Based on: https://scottsuhy.com/2021/02/01/image-alignment-feature-based-in-opencv-js-javascript/

  let detector_option = 2;  //KAZE document.getElementById('detector').value;
  let match_option = 1;  //knnMatch document.getElementById('match').value;
  let matchDistance_option = 20;  //document.getElementById('distance').value;
  let knnDistance_option = 0.7;  //document.getElementById('knn_distance').value;
  let pyrDown_option = "No";  //document.getElementById('pyrDown').value;

//   //If the users is going to try a second attempt we need to clear out the canvases
//   let image_blank_element = document.getElementById('image_blank');
//   let im_blank = cv.imread(image_blank_element);
//   cv.imshow('keypoints1', im_blank);
//   cv.imshow('keypoints2', im_blank);
//   cv.imshow('imageCompareMatches', im_blank);
//   cv.imshow('image_Aligned', im_blank);
//   cv.imshow('inlierMatches', im_blank);

  console.error("STEP 1: READ IN IMAGES **********************************************************************");
  //im2 is the original reference image we are trying to align to
  let im2 = cv.imread('image_A_element_id');
  getMatStats(im2, "original reference image");
  //im1 is the image we are trying to line up correctly
  
  let resultSize = im2.size();
  inputImageB = createImage(resultSize.width, resultSize.height);
  cvMatToP5Image(im2, inputImageB);
  
  let im1 = cv.imread(image_B_element);
  
  resultSize = im1.size();
  inputImageA = createImage(resultSize.width, resultSize.height);
  cvMatToP5Image(im1, inputImageA);

  getMatStats(im1, "original image to line up");

  if (pyrDown_option !== 'No') {
      console.log("User selected option to pyrDown image");
      cv.pyrDown(im1, im1, new cv.Size(0, 0), cv.BORDER_DEFAULT);
      cv.pyrDown(im2, im2, new cv.Size(0, 0), cv.BORDER_DEFAULT);
      getMatStats(im1, "new stats for im1");
      getMatStats(im2, "new stats for im2");
  }

  console.error("STEP 2: CONVERT IMAGES TO GRAYSCALE *********************************************************");
  //17            Convert images to grayscale
  //18            Mat im1Gray, im2Gray;
  //19            cvtColor(im1, im1Gray, CV_BGR2GRAY);
  //20            cvtColor(im2, im2Gray, CV_BGR2GRAY);
  let im1Gray = new cv.Mat();
  let im2Gray = new cv.Mat();
  cv.cvtColor(im1, im1Gray, cv.COLOR_BGRA2GRAY);
  getMatStats(im1Gray, "reference image converted to BGRA2GRAY");
  cv.cvtColor(im2, im2Gray, cv.COLOR_BGRA2GRAY);
  getMatStats(im2Gray, "image to line up converted to BGRA2GRAY");

  console.error("STEP 3: DETECT FEATURES & COMPUTE DESCRIPTORS************************************************");
  //22            Variables to store keypoints and descriptors
  //23            std::vector<KeyPoint> keypoints1, keypoints2;
  //24            Mat descriptors1, descriptors2;
  let keypoints1 = new cv.KeyPointVector();
  let keypoints2 = new cv.KeyPointVector();
  let descriptors1 = new cv.Mat();
  let descriptors2 = new cv.Mat();
  //26            Detect ORB features and compute descriptors.
  //27            Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  //28            orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
  //29            orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

  if (detector_option == 0) {
      var X = new cv.ORB(5000);
      console.log("using cv.ORB");
  } else if (detector_option == 1) {
      var X = new cv.AKAZE();
      console.log("using cv.AKAZE");
  } else if (detector_option == 2) {
      var X = new cv.KAZE();
      console.log("using cv.KAZE");
  }

  X.detectAndCompute(im1Gray, new cv.Mat(), keypoints1, descriptors1);
  X.detectAndCompute(im2Gray, new cv.Mat(), keypoints2, descriptors2);

  console.log("keypoints1: ", keypoints1);
  console.log("descriptors1: ", descriptors1);
  console.log("keypoints2: ", keypoints2);
  console.log("descriptors2: ", descriptors2);
  getMatStats(descriptors1, "descriptors1");
  getMatStats(descriptors2, "descriptors2");

//   //draw all the keypoints on each image and display it to the user
//   let keypoints1_img = new cv.Mat();
//   let keypoints2_img = new cv.Mat();
//   let keypointcolor = new cv.Scalar(0,255,0, 255);
//   //this flag does not work because of bug https://github.com/opencv/opencv/issues/13641?_pjax=%23js-repo-pjax-container
//   cv.drawKeypoints(im1Gray, keypoints1, keypoints1_img, keypointcolor);
//   cv.drawKeypoints(im2Gray, keypoints2, keypoints2_img, keypointcolor); //cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);//,

//   cv.imshow('keypoints1', keypoints1_img);
//   cv.imshow('keypoints2', keypoints2_img);

  // use to debug and list out all the keypoints
  console.log("there are a total of ", keypoints1.size(), " keypoints1 (img to aligned) and ", keypoints2.size(), " keypoints2 (reference)");
  console.log("here are the first 5 keypoints for keypoints1 - image to align.");
  for (let i = 0; i < keypoints1.size(); i++) {
      console.log("keypoints1: [",i,"]", keypoints1.get(i).pt.x, keypoints1.get(i).pt.y);
      if (i === 5){break;}
  }

  console.log("here are the first 5 keypoints for keypoints2 -- reference image");
  for (let i = 0; i < keypoints2.size(); i++) {
      console.log("keypoints2: [",i,"]", keypoints2.get(i).pt.x, keypoints2.get(i).pt.y);
      if (i === 5){break;}
  }

  console.log("there are a total of [", descriptors1.cols, "][", descriptors1.rows, "] descriptors1 [cols][rows] (img to aligned) and [", descriptors2.cols, "][", descriptors2.rows, "] descriptors2 (reference) [cols][rows]");

  console.error("STEP 4: MATCH FEATURES **********************************************************************");
  //31            Match features.
  //32            std::vector<DMatch> matches;
  //33            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //34            matcher->match(descriptors1, descriptors2, matches, Mat());

  let good_matches = new cv.DMatchVector();

  if(match_option == 0){//match
      console.log("using match...");
      let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
      let matches = new cv.DMatchVector();
      bf.match(descriptors1, descriptors2, matches);

      //36            Sort matches by score
      //37            std::sort(matches.begin(), matches.end());
      //39            Remove not so good matches
      //40            const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
      //41            matches.erase(matches.begin()+numGoodMatches, matches.end());
      console.log("matches.size: ", matches.size());
      for (let i = 0; i < matches.size(); i++) {
          if (matches.get(i).distance < matchDistance_option) {
              good_matches.push_back(matches.get(i));
          }
      }
      if(good_matches.size() <= 3){
          alert("Less than 4 good matches found! counter =" + good_matches.size() + " try changing distance.");
          return;
      }
  }
  else if(match_option == 1) { //knnMatch
      console.log("using knnMatch...");
      let bf = new cv.BFMatcher();
      let matches = new cv.DMatchVectorVector();
      //Reference: https://docs.opencv.org/3.3.0/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89
      bf.knnMatch(descriptors1, descriptors2, matches, 2);

      let counter = 0;
      for (let i = 0; i < matches.size(); ++i) {
          let match = matches.get(i);
          let dMatch1 = match.get(0);
          let dMatch2 = match.get(1);
          //console.log("[", i, "] ", "dMatch1: ", dMatch1, "dMatch2: ", dMatch2);
          if (dMatch1.distance <= dMatch2.distance * parseFloat(knnDistance_option)) {
              //console.log("***Good Match***", "dMatch1.distance: ", dMatch1.distance, "was less than or = to: ", "dMatch2.distance * parseFloat(knnDistance_option)", dMatch2.distance * parseFloat(knnDistance_option), "dMatch2.distance: ", dMatch2.distance, "knnDistance", knnDistance_option);
              good_matches.push_back(dMatch1);
              counter++;
          }
      }
      if(counter <= 3){
          alert("Less than 4 good matches found! Counter=" + counter + " try changing distance %. It's currently " + knnDistance_option);
          return;
      }
      console.log("keeping ", counter, " points in good_matches vector out of ", matches.size(), " contained in this match vector:", matches);
      console.log("here are first 5 matches");
      for (let t = 0; t < matches.size(); ++t) {
          console.log("[" + t + "]", "matches: ", matches.get(t));
          if (t === 5){break;}
      }
  }
  console.log("here are first 5 good_matches");
  for (let r = 0; r < good_matches.size(); ++r) {
      console.log("[" + r + "]", "good_matches: ", good_matches.get(r));
      if (r === 5){break;}
  }

  console.error("STEP 5: DRAW TOP MATCHES AND OUTPUT IMAGE TO SCREEN ***************************************");
  //44            Draw top matches
  //45            Mat imMatches;
  //46            drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
  //47            imwrite("matches.jpg", imMatches);
  let imMatches = new cv.Mat();
  let color = new cv.Scalar(0,255,0, 255);
  //cv.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, imMatches, color);
  //cv.imshow('imageCompareMatches', imMatches);
  getMatStats(imMatches, "imMatches");

  console.error("STEP 6: EXTRACT LOCATION OF GOOD MATCHES AND BUILD POINT1 and POINT2 ARRAYS ***************");
  //50            Extract location of good matches
  //51            std::vector<Point2f> points1, points2;
  //53            for( size_t i = 0; i < matches.size(); i++ )
  //54            {
  //55                points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
  //56                points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  //57            }

  //this is a test
  //let points1 = create2dPointsArray(good_matches.size(), 2, 0);
  //let points2 = create2dPointsArray(good_matches.size(), 2, 0);
  
  for (let i = 0; i < good_matches.size(); i++) {

      points1.push(keypoints1.get(good_matches.get(i).queryIdx ).pt.x );
      points1.push(keypoints1.get(good_matches.get(i).queryIdx ).pt.y );
      points2.push(keypoints2.get(good_matches.get(i).trainIdx ).pt.x );
      points2.push(keypoints2.get(good_matches.get(i).trainIdx ).pt.y );

      //this is a test
      // points1[i][0] = keypoints1.get(good_matches.get(i).queryIdx ).pt.x;
      // points1[i][1] = keypoints1.get(good_matches.get(i).queryIdx ).pt.y;
      // points2[i][0] = keypoints2.get(good_matches.get(i).trainIdx ).pt.x;
      // points2[i][1] = keypoints2.get(good_matches.get(i).trainIdx ).pt.y;

      //from: https://answers.opencv.org/question/235594/opencvjs-findperspective-returns-wrong-corners-coordinates/
      //points1.push_back( new cv.Point(keypoints1.get(good_matches.get(i).queryIdx).pt.x, keypoints1.get(good_matches.get(i).queryIdx).pt.y));
      //points2.push_back( new cv.Point(keypoints2.get(good_matches.get(i).trainIdx).pt.x, keypoints2.get(good_matches.get(i).trainIdx).pt.y));
  }
  console.log("points1:", points1,"points2:", points2);

  console.error("STEP 7: CREATE MAT1 and MAT2 FROM POINT1 and POINT2 ARRAYS ********************************");
  //Alternative:
  //let mat1 = cv.matFromArray(points1.length, 1, cv.CV_32FC2, points1);
  //let mat2 = cv.matFromArray(points2.length, 1, cv.CV_32FC2, points2);

  var mat1 = new cv.Mat(points1.length,1,cv.CV_32FC2);
  mat1.data32F.set(points1);
  var mat2 = new cv.Mat(points2.length,1,cv.CV_32FC2);
  mat2.data32F.set(points2);

  // this is a test
  // var mat1 = new cv.Mat(points1.length,2,cv.CV_32F);
  // mat1.data32F.set(points1);
  // var mat2 = new cv.Mat(points2.length,2,cv.CV_32F);
  // mat2.data32F.set(points2);

  getMatStats(mat1, "mat1 prior to homography");
  getMatStats(mat2, "mat2 prior to homography");

  console.error("STEP 8: CALCULATE HOMOGRAPHY USING MAT1 and MAT2 ******************************************");
  //59            Find homography
  //60            h = findHomography( points1, points2, RANSAC );
  //Reference: https://docs.opencv.org/3.3.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
  //mat1:	Coordinates of the points in the original plane, a matrix of the type CV_32FC2 or vector<Point2f> .
  //mat2:	Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or a vector<Point2f> .

  let findHomographyMask = new cv.Mat();//test
  h = cv.findHomography(mat1, mat2, cv.RANSAC, 3, findHomographyMask);
  if (h.empty())
  {
      alert("homography matrix empty!");
      return;
  }
  else{
      console.log("h:", h);
      console.log("[", h.data64F[0],",", h.data64F[1], ",",h.data64F[2]);
      console.log("", h.data64F[3],",", h.data64F[4], ",", h.data64F[5]);
      console.log("", h.data64F[6],",", h.data64F[7], ",", h.data64F[8], "]");

      getMatStats(findHomographyMask, "findHomographyMask"); //test
      console.log("here are the inliers from RANSAC, compare to the good_matches array above", findHomographyMask.rows);//test
      //for (let i = 0; i < findHomographyMask.rows; ++i) {
      //    console.log("inliers", findHomographyMask.data[i], "points2: ", points2[i]);
      //}
      good_inlier_matches = new cv.DMatchVector();
      for (let i = 0; i < findHomographyMask.rows; i=i+2) {
          if(findHomographyMask.data[i] === 1 || findHomographyMask.data[i+1] === 1) {
              let x = points2[i];
              let y = points2[i + 1];
              //console.log("i: ", i, " x: ", x, " y: ", y, "   Found it in points2!");s
              for (let j = 0; j < keypoints2.size(); ++j) {
                  if (x === keypoints2.get(j).pt.x && y === keypoints2.get(j).pt.y) {
                      //console.log("  -- j: ", j, "    Found item in keypoints2!")
                      for (let k = 0; k < good_matches.size(); ++k) {
                          if (j === good_matches.get(k).trainIdx) {
                              //console.log("  -- k: ", k, "    Found item in good_matches!")
                              good_inlier_matches.push_back(good_matches.get(k));
                          }
                      }
                  }
              }
          }
      }
      var inlierMatches = new cv.Mat();
      //cv.drawMatches(im1, keypoints1, im2, keypoints2, good_inlier_matches, inlierMatches, color);
      //cv.imshow('inlierMatches', inlierMatches);
      console.log("Good Matches: ", good_matches.size(), " inlier Matches: ", good_inlier_matches.size());

      console.log("here are inlier good_matches");
      for (let r = 0; r < good_inlier_matches.size(); ++r) {
          console.log("[" + r + "]", "good_inlier_matches: ", good_inlier_matches.get(r));
          //console.log(keypoints1[good_inlier_matches.get(r).queryIdx], keypoints2[good_inlier_matches.get(r).trainIdx]);
      }
      // console.log("here are outlier good_matches (better said, BAD Matches)");
      // for (let r = 0; r < bad_outlier_matches.size(); ++r) {
      //     console.log("[" + r + "]", "good_outlier_matches: ", bad_outlier_matches.get(r));
      // }
      
      let src = cv.matFromArray(3, 1, cv.CV_32FC2, [0,0,1]);
      getMatStats(src, "src");
    
//       src(0,0)=p.x; 
//       src(1,0)=p.y; 
//       src(2,0)=1.0; 

//       cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
  }
  getMatStats(findHomographyMask, "findHomographyMask");

  console.error("STEP 9: WARP IMAGE TO ALIGN WITH REFERENCE **************************************************");
  //62          Use homography to warp image
  //63          warpPerspective(im1, im1Reg, h, im2.size());
  //Reference: https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
  /* DJC
  let image_B_final_result = new cv.Mat();
  cv.warpPerspective(im1, image_B_final_result, h, im2.size());
  //cv.imshow('image_Aligned', image_B_final_result);
  getMatStats(image_B_final_result, "finalMat");
  */

  //X.delete();
  descriptors1.delete();
  descriptors2.delete();
  keypoints1.delete();
  keypoints2.delete();
  im1Gray.delete();
  im2Gray.delete();
  //h.delete();
  // image_B_final_result may not have been created (warpPerspective is commented out).
  // Guard deletion to avoid ReferenceError.
  if (typeof image_B_final_result !== 'undefined' && image_B_final_result !== null) {
    try { image_B_final_result.delete(); } catch (e) { console.warn('failed to delete image_B_final_result:', e); }
  }
  mat1.delete();
  mat2.delete();
  //inlierMatches.delete();
}

function getMatStats(Mat, name)
{
let type = Mat.type()
let channels = Mat.channels();
let cols = Mat.cols;
let rows = Mat.rows;
let depth = Mat.depth();
let baseline_colorspace = "";
let baseline_matType = "";

if (channels == 4){
baseline_colorspace = "RGBA or BGRA"
if(type == 24){baseline_matType = "CV_8UC4";}
if(type == 25){baseline_matType = "CV_8SC4";}
if(type == 26){baseline_matType = "CV_16UC4";}
if(type == 27){baseline_matType = "CV_16SC4";}
if(type == 28){baseline_matType = "CV_32SC4";}
if(type == 29){baseline_matType = "CV_32FC4";}
if(type == 30){baseline_matType = "CV_64FC4";}
}
if (channels == 3){
baseline_colorspace = "RGB, HSV or BGR";
if(type == 16){baseline_matType = "CV_8UC3";}
if(type == 17){baseline_matType = "CV_8SC3";}
if(type == 18){baseline_matType = "CV_16UC3";}
if(type == 19){baseline_matType = "CV_16SC3";}
if(type == 20){baseline_matType = "CV_32SC3";}
if(type == 21){baseline_matType = "CV_32FC3";}
if(type == 22){baseline_matType = "CV_64FC3";}
}
if (channels == 2){
baseline_colorspace = "unknown"
if(type == 8){baseline_matType = "CV_8UC2";}
if(type == 9){baseline_matType = "CV_8SC2";}
if(type == 10){baseline_matType = "CV_16UC2";}
if(type == 11){baseline_matType = "CV_16SC2";}
if(type == 12){baseline_matType = "CV_32SC2";}
if(type == 13){baseline_matType = "CV_32FC2";}
if(type == 14){baseline_matType = "CV_64FC2";}
}
if (channels == 1){
baseline_colorspace = "GRAY"
if(type == 0){baseline_matType = "CV_8UC1";}
if(type == 1){baseline_matType = "CV_8SC1";}
if(type == 2){baseline_matType = "CV_16UC1";}
if(type == 3){baseline_matType = "CV_16SC1";}
if(type == 4){baseline_matType = "CV_32SC1";}
if(type == 5){baseline_matType = "CV_32FC1";}
if(type == 6){baseline_matType = "CV_64FC1";}
}

console.log("MatName :(" + name + ") ", Mat);
console.log("   MatStats:channels=" + channels + " type:" + type + " cols:" + cols + " rows:" + rows );
console.log("   depth:" + depth + " colorspace:" + baseline_colorspace + " type:" + baseline_matType );

return;
}

function cvMatToP5Image(mat, image) {
  //mat to canvas
  let tempCanvas = document.createElement('canvas');
  tempCanvas.id = 'tempCanvas';
  tempCanvas.classList.add('hide');
  document.body.appendChild(tempCanvas);
  
  let resultSize = mat.size();
  tempCanvas.width = resultSize.width;
  tempCanvas.height = resultSize.height
  
  cv.imshow('tempCanvas', mat);
  
  canvasToP5Image(tempCanvas, image);
  tempCanvas.remove();
}

function drawImageWithHomography(img, H_flat) {
  fill(0);
  rect(0,0,100,100);
  
  if (!img || H_flat.length !== 9) return; // Ensure valid input

  let H = [
    [H_flat[0], H_flat[1], H_flat[2]],
    [H_flat[3], H_flat[4], H_flat[5]],
    [H_flat[6], H_flat[7], H_flat[8]]
  ];

  let w = img.width / 2, h = img.height / 2;

  // Define original image corners
  let srcCorners = [
    [-w, -h], [w, -h], [w, h], [-w, h]
  ];

  // Apply homography to get new corners
  let dstCorners = srcCorners.map(([x, y]) => applyHomography(x, y, H));

  //console.log('dstCorners', dstCorners);
  
  //texture(img);
  beginShape();
  for (let i = 0; i < 4; i++) {
    let [x, y] = dstCorners[i];
    let u = i % 2, v = Math.floor(i / 2); // Texture coordinates (0 or 1)
    vertex(x, y, 0, u, v);
  }
  endShape(CLOSE);
}

// Function to apply homography matrix
function applyHomography(x, y, H) {
  let newX = H[0][0] * x + H[0][1] * y + H[0][2];
  let newY = H[1][0] * x + H[1][1] * y + H[1][2];
  let newW = H[2][0] * x + H[2][1] * y + H[2][2]; // Homogeneous coordinate

  return [newX / newW, newY / newW]; // Normalize
}