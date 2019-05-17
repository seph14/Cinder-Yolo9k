//
//  YoloHandler.cpp
//  AIVideoMaker
//
//  Created by Seph Li on 08/08/2018.
//

#include "YoloHandler.h"

// yolo models
#include "YOLO9000.h"

#include "Tree.h"
#import  <CoreML/CoreML.h>
#import  <Vision/Vision.h>

using namespace std;
using namespace ci;
using namespace ml;

static const int INPUT_SIZE_9k              = 544;
static const NSInteger maxBoundingBoxes9k   = 30;

static const float anchors_9k[]             = { 0.77871f, 1.14074f, 3.00525f, 4.31277f, 9.22725f, 9.61974f };
static const float confidenceThreshold9k    = 0.095f, iouThreshold9k = 0.35f;

API_AVAILABLE(macos(10.13)) MLModel         *_model9k;
API_AVAILABLE(macos(10.13)) VNCoreMLModel   *_mlModel9k;
API_AVAILABLE(macos(10.13)) VNCoreMLRequest *_mlRequest9k;
NSArray                                     *_rqArray9k, *_observation9k;
NSDictionary                                *_rqDictionary9k;

std::vector<ciPrediction> mResult9k;
tree*                     mTree;

/*------------------------------------------------------------------------------------*/
// helper functions
ci::ivec2 Yolo9kHandler::getExpectedSize(){ return ivec2(INPUT_SIZE_9k, INPUT_SIZE_9k); }

void Yolo9kHandler::loadTreeFile(ci::DataSourceRef data){
    if(mTree != nullptr) free(mTree);
    mTree = read_tree(data);
}

void Yolo9kHandler::printTreeStructure( const std::vector<std::string>& classNames ){
    if(mTree == nullptr) return;
    
    for(int i = 0; i < mTree->group_size[0]; ++i){
        int idx = i + mTree->group_offset[0];
        app::console() << classNames[idx] << endl;
    }
    
}

float Yolo9kHandler::calcSimilarity( int objectA, int objectB ){
    if(objectA == objectB) return 1.f;

    if(objectB < objectA){
        int tmp = objectB;
        objectB = objectA;
        objectA = tmp;
    }
    
    const int MAX_LEVEL = 12;
    int level           = 1;
    int tmpB            = objectB;
    
    while(level < MAX_LEVEL){
        if(mTree->parent[tmpB] >= 0){
            int parent = mTree->parent[tmpB];
            if(parent == objectA) return 1.f - (float)level / MAX_LEVEL;
            tmpB = parent;
            level ++;
        }else break;
    }
    
    if( mTree->parent[objectA] < 0 ) return 0.f;
    
    level    = 1;
    int tmpA = objectA;
    tmpB     = objectB;
    while (level < MAX_LEVEL) {
        
        auto pp = mTree->parent[tmpA];
        if(pp >= 0) tmpA = pp;
        else break;
        
        int steps = MAX_LEVEL - level;
        
        while(level < steps){
            if(mTree->parent[tmpB] >= 0){
                int parent = mTree->parent[tmpB];
                if(parent == tmpA)
                    return 1.f - (float)level / MAX_LEVEL;
                tmpB = parent;
                level ++;
            }else break;
        }
        
        level = MAX_LEVEL - steps + 1;
    }
    
    level   = 1;
    tmpA    = objectA;
    tmpB    = objectB;
    while (level < MAX_LEVEL) {
        
        auto pp = mTree->parent[tmpB];
        if(pp >= 0) tmpB = pp;
        else break;
        
        int steps = MAX_LEVEL - level;
        
        while(level < steps){
            if(mTree->parent[tmpA] >= 0){
                int parent = mTree->parent[tmpA];
                if(parent == tmpB)
                    return 1.f - (float)level / MAX_LEVEL;
                tmpA = parent;
                level ++;
            }else break;
        }
        
        level = MAX_LEVEL - steps + 1;
    }
    
    return 0.f;
}

/*------------------------------------------------------------------------------------*/
// math functions
int offset(int channel, int x, int y, int channelStride, int xStride, int yStride){
    return channel*channelStride + y*yStride + x*xStride;
}
float sigmoid(float z){ return 1.f / (1.f + (float)exp(-z)); }
void softmax(float vals[], int count) {
    float max = - FLT_MAX;
    for (int i=0; i<count; i++) { max = fmax(max, vals[i]); }
    float sum = 0.0;
    for (int i=0; i<count; i++) { vals[i] = exp(vals[i] - max); sum += vals[i]; }
    for (int i=0; i<count; i++) { vals[i] /= sum; }
}
bool sortConfidence (Prediction i, Prediction j) {
    return (i.score>j.score);
}
/**
 Computes intersection-over-union overlap between two bounding boxes.
 */
float IOU(CGRect a, CGRect b) {
    float areaA = a.size.width * a.size.height;
    if(areaA <= 0) { return 0; }
    
    float areaB = b.size.width * b.size.height;
    if(areaB <= 0) { return 0; }
    
    float aminX = a.origin.x - a.size.width/2.f;
    float amaxX = a.origin.x + a.size.width/2.f;
    float aminY = a.origin.y - a.size.height/2.f;
    float amaxY = a.origin.y + a.size.height/2.f;
    
    float bminX = b.origin.x - b.size.width/2.f;
    float bmaxX = b.origin.x + b.size.width/2.f;
    float bminY = b.origin.y - b.size.height/2.f;
    float bmaxY = b.origin.y + b.size.height/2.f;
    
    float intersectionMinX = fmax(aminX, bminX);
    float intersectionMinY = fmax(aminY, bminY);
    float intersectionMaxX = fmin(amaxX, bmaxX);
    float intersectionMaxY = fmin(amaxY, bmaxY);
    float intersectionArea = fmax(intersectionMaxY - intersectionMinY, 0) * fmax(intersectionMaxX - intersectionMinX, 0);
    return float(intersectionArea / (areaA + areaB - intersectionArea));
}
/**
 Removes bounding boxes that overlap too much with other boxes that have
 a higher score.
 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
 - Parameters:
 - boxes: an array of bounding boxes and their scores
 - limit: the maximum number of boxes that will be selected
 - threshold: used to decide whether boxes overlap too much
 */
std::vector<Prediction> nonMaxSuppression(std::vector<Prediction> boxes, NSInteger limit, float threshold) {
    if(boxes.size() <= 0) return boxes;
    
    // Do an argsort on the confidence scores, from high to low.
    std::sort(boxes.begin(), boxes.end(), sortConfidence);
    
    std::vector<Prediction>   selected;
    std::vector<bool> active( boxes.size(), true );
    int numActive       = (int)boxes.size();
    bool shouldBreak    = false;
    
    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    for( int i = 0; i < boxes.size(); i++ ) {
        if(shouldBreak) break;
        if(active[i]) {
            auto boxA = boxes[i];
            selected.push_back(boxA);
            if(selected.size() >= limit) { break; }
            
            for( int j = i+1; j < boxes.size(); j++ ) {
                if(active[j]) {
                    auto boxB = boxes[j];
                    if(IOU(boxA.rect, boxB.rect) > threshold){
                        active[j] = false;
                        numActive -= 1;
                        if(numActive <= 0) { shouldBreak = true; break; }
                    }
                }
            }
        }
    }
    return selected;
}
/*------------------------------------------------------------------------------------*/
// predict function

API_AVAILABLE(macos(10.13))
std::vector<Prediction> computeBoundingBoxes9k(MLMultiArray* features){
    assert(features.count == (9418+5)*3*17*17);
    
    static const int blockSize    = 32;
    static const int gridHeight   = 17;
    static const int gridWidth    = 17;
    static const int boxesPerCell = 3;
    static const int numClasses   = 9418;
    
    std::vector<Prediction> predictions;
    
    double* featurePointer  = (double*)features.dataPointer;
    int channelStride       = features.strides[0].intValue;
    int yStride             = features.strides[1].intValue;
    int xStride             = features.strides[2].intValue;
    
    for(int cy = 0; cy < gridHeight; cy++) {
        for(int cx = 0; cx < gridWidth; cx++) {
            for (int b = 0; b < boxesPerCell; b++) {
                int channel     = b * (numClasses + 5);
                int obj_index   = offset(channel + 4, cx, cy, channelStride, xStride, yStride);
                float scale     = sigmoid(float(featurePointer[obj_index]));
                int class_index = offset(channel + 5, cx, cy, channelStride, xStride, yStride);
                
                float tx = float(featurePointer[offset(channel + 0, cx, cy, channelStride, xStride, yStride)]);
                float ty = float(featurePointer[offset(channel + 1, cx, cy, channelStride, xStride, yStride)]);
                float tw = float(featurePointer[offset(channel + 2, cx, cy, channelStride, xStride, yStride)]);
                float th = float(featurePointer[offset(channel + 3, cx, cy, channelStride, xStride, yStride)]);
                
                float x = (float(cx) + sigmoid(tx)) * blockSize;
                float y = (float(cy) + sigmoid(ty)) * blockSize;
                float w = (float)exp(tw) * anchors_9k[2*b    ] * blockSize;
                float h = (float)exp(th) * anchors_9k[2*b + 1] * blockSize;
                
                hierarchy_predictions(featurePointer + class_index, numClasses, mTree, 0,     gridWidth*gridHeight);
                int j =  hierarchy_top_prediction(featurePointer + class_index, mTree, 0.25f, gridWidth*gridHeight);
                
                if(scale > confidenceThreshold9k){
                    CGRect rect =
                    CGRectMake( CGFloat(x - w/2), CGFloat(y - h/2), CGFloat(w), CGFloat(h) );
                    
                    Prediction prediction;
                    prediction.classIndex   = j;
                    prediction.score        = scale;
                    prediction.rect         = rect;
                    predictions.push_back(prediction);
                }
            }
        }
    }
    return nonMaxSuppression(predictions, maxBoundingBoxes9k, iouThreshold9k);
}
/*------------------------------------------------------------------------------------*/
// 9k logic

void Yolo9kHandler::init(){
    if (@available(macOS 10.13, *)) {
        _model9k      = [[[YOLO9000 alloc] init] model];
        _mlModel9k    = [VNCoreMLModel modelForMLModel:_model9k error:nil];
        _mlRequest9k  = [[VNCoreMLRequest alloc] initWithModel:_mlModel9k completionHandler:(VNRequestCompletionHandler) ^(VNRequest *request, NSError *error){
            if(error != nil){
                NSString *err = error.localizedDescription;
                app::console() << "Yolo9k returned with error:" << [err UTF8String] << endl;
                return;
            }
                       
            _observation9k                              = request.results; //[request.results copy];
            VNCoreMLFeatureValueObservation *topRequest = ((VNCoreMLFeatureValueObservation *)(_observation9k[0]));
            MLMultiArray *features                      = topRequest.featureValue.multiArrayValue;
            std::vector<Prediction> boundingboxes       = computeBoundingBoxes9k(features);
            if(boundingboxes.size() > 0){
                float size = (float)INPUT_SIZE_9k;
                for(auto box : boundingboxes){
                    ciPrediction elem;
                    elem.classIndex = box.classIndex;
                    elem.score      = box.score;
                       
                    float x0        = fmax(0.f, box.rect.origin.x) / size;
                    float y0        = fmax(0.f, box.rect.origin.y) / size;
                    float w         = (float)box.rect.size.width / size;
                    float h         = (float)box.rect.size.height / size;
                    elem.rect       = Rectf(x0, y0, fmin(1.f, x0+w), fmin(1.f,y0+h));
                    mResult9k.push_back(elem);
                }
            }
            //[_observation9k release];
        }];
    }
    _rqDictionary9k = [[NSDictionary alloc] init];
}

std::vector<ciPrediction> Yolo9kHandler::process( ci::Surface8u surf ){
    if (@available(macOS 10.13, *)) {
        uint width                      = surf.getWidth();
        uint height                     = surf.getHeight();
        unsigned char *pixels           = (unsigned char*)malloc(height*width*4);
        CGColorSpaceRef colorSpaceRef   = CGColorSpaceCreateDeviceRGB();
        CGContextRef context            = CGBitmapContextCreate
        (pixels, width, height, 8, 4*width,
         colorSpaceRef, kCGImageAlphaPremultipliedLast);
        CGColorSpaceRelease(colorSpaceRef);
    
        auto pixelIter = surf.getIter();
        int x = 0, y = 0;
        while( pixelIter.line() ) {
            x = 0;
            while( pixelIter.pixel() ) {
                int idx         = (width*y+x)*4;
                pixels[idx+0]   = pixelIter.r();
                pixels[idx+1]   = pixelIter.g();
                pixels[idx+2]   = pixelIter.b();
                pixels[idx+3]   = pixelIter.a();
                x ++;
            }
            y ++;
        }
    
        auto imageRef = CGBitmapContextCreateImage(context);
        CGContextRelease(context);
        free(pixels);
    
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
                                          initWithCGImage:imageRef
                                          options:        _rqDictionary9k];
        
        mResult9k.clear();
        _rqArray9k = @[_mlRequest9k];
        [handler performRequests:_rqArray9k error:nil];
        [_rqArray9k release];
        [handler release];
        CGImageRelease(imageRef);
    }
    return mResult9k;
}
