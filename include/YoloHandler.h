//
//  Yolo3Handler.h
//  Analyzer
//
//  Created by Seph Li on 15/04/2019.
//

#ifndef Yolo3Handler_h
#define Yolo3Handler_h

#include "stdlib.h"
#include <vector>

namespace ml{
    typedef struct Prediction {
        int   classIndex;
        float score;
        CGRect rect;
    } Prediction;
    
    typedef struct ciPrediction{
        int         classIndex;
        float       score;
        ci::Rectf   rect;
    } ciPrediction;
    
    class Yolo9kHandler {
    public:
        static void     init();
        static void     loadTreeFile(ci::DataSourceRef data);
        static void     printTreeStructure( const std::vector<std::string>& classNames );
        static float    calcSimilarity( int objectA, int objectB );
        
        static ci::ivec2 getExpectedSize();
        static std::vector<ciPrediction> process( ci::Surface8u surf );
    };
};

#endif /* Yolo3Handler_h */
