import ocv.*;
import processing.video.*;

import org.opencv.video.*;
import org.opencv.core.*;
import org.opencv.calib3d.*;
import org.opencv.contrib.*;
import org.opencv.objdetect.*;
import org.opencv.imgproc.*;
import org.opencv.utils.*;
import org.opencv.features2d.*;
import org.opencv.highgui.*;
import org.opencv.ml.*;
import org.opencv.photo.*;

import java.util.Vector;

PImage pimg;
PImage pimg2;
PImage pimg3;
Capture cam;
ocvP5 ocv;
CascadeClassifier classifier;

ArrayList<Rect> faceRects;

void setup()
{
  System.load(new File("/opt/local/share/OpenCV/java/libopencv_java2410.dylib")
  .getAbsolutePath());

  ocv = new ocvP5(this);  
  size(1280, 960);

  String[] cameras = Capture.list();
  cam = new Capture(this, cameras[0]);
  cam.start();
  
  classifier = new CascadeClassifier(dataPath("haarcascade_frontalface_default.xml"));
  
  //pimg2 = loadImage("img1.png");
  pimg3 = loadImage("img2.png");
  faceRects = new ArrayList(); 

}

void draw() 
{

  if (cam.available() == true) 
  {
    cam.read();
    pimg = cam;
    Mat m = ocv.toCV(pimg);

    Mat gray = new Mat(m.rows(), m.cols(), CvType.CV_8U);
    Imgproc.cvtColor(m, gray, Imgproc.COLOR_BGRA2GRAY);

     MatOfRect objects = new MatOfRect();

    Size minSize = new Size(350, 350);
    Size maxSize = new Size(500, 500);

    classifier.detectMultiScale(gray, objects, 1.1, 3, 
    Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_DO_ROUGH_SEARCH, 
    minSize, maxSize);

    faceRects.clear();

    for (Rect rect: objects.toArray()) {
      faceRects.add(new Rect(rect.x, rect.y, rect.width, rect.height));
    }
  }

  image(cam, 0, 0);

  for (int i = 0; i < faceRects.size(); i++) {
    /*image(pimg2, faceRects.get(i).x, faceRects.get(i).y, faceRects.get(i).width, 
    faceRects.get(i).height);*/
    image(pimg3, faceRects.get(i).x, faceRects.get(i).y, faceRects.get(i).width, 
    faceRects.get(i).height);
  }
}
