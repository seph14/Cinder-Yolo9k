# Cinder-Yolo9k
Simple app showing how to use Yolo9k model with CoreML in Cinder;

This repo shows how to use Yolo9k model with CoreML and pass in Cinder's surface from a camera feed as input.
The possibility result per each class seems to change a lot after converted to CoreML format but the results are looking valid to me.

# Download Yolo9k Model
The converted Yolo9K model can be downloaded [here](https://drive.google.com/file/d/1KNRW3wUqQFuJUwW6cv8yTVrPi7bugn6H/view?usp=sharing).
And then put the unziped mlmodel file in the include folder - it will be automatically recognized in Xcode.

This project is built with Cinder 0.9.1
