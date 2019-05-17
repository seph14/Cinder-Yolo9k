#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "Resources.h"

#include "cinder/Capture.h"
#include "cinder/ip/Resize.h"
#include "cinder/Text.h"
#include "cinder/Rand.h"
#include "cinder/gl/TextureFont.h"
#include "cinder/Utilities.h"
#include "YoloHandler.h"

using namespace ci;
using namespace ci::app;
using namespace std;

std::mutex mtx9k;

class Yolo9KTestApp : public App {
  public:
    vector<ml::ciPrediction> mRes9k;
    CaptureRef               mCapture;
    Surface8uRef             mFeed, mFeed9k;
    gl::Texture2dRef         mTex;
    vector<string>           mClass;
    
    Font                mDisplayFont;
    gl::TextureFontRef  mDisplayTex;
    bool                mShouldQuit, mFeed9Updated;
    shared_ptr<thread>  mThread9k;
    
    void predictImage9k();
    
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
    void cleanup()  override;
};

void Yolo9KTestApp::cleanup() {
    mShouldQuit = true;
    mCapture->stop();
    mThread9k->join();
}

void Yolo9KTestApp::setup() {
    setFrameRate(30.f);
    gl::enableVerticalSync(false);
    
    auto file   = loadResource(RES_YOLO_NAMES);
    auto stream = file->createStream();
    while(!stream->isEof()) mClass.push_back(stream->readLine());
    ml::Yolo9kHandler::init();
    ml::Yolo9kHandler::loadTreeFile(loadResource(RES_YOLO_TREE));
    
    mDisplayFont    = Font( "Helvetica-Light", 12 );
    mDisplayTex     = gl::TextureFont::create( mDisplayFont );
    
    mFeed9Updated   = false;
    mShouldQuit     = false;
    
    try {
        mCapture = Capture::create( 960, 540 );
        mCapture->start();
    } catch( ci::Exception &exc ) {
        console() << "Failed to init capture " << exc.what() << std::endl;
    }
    
    mThread9k   = shared_ptr<thread>(new thread(bind(&Yolo9KTestApp::predictImage9k, this)));
}

void Yolo9KTestApp::predictImage9k() {
    ci::ThreadSetup threadSetup;
    
    while ( !mShouldQuit ) {
        if(mFeed9Updated){
            lock_guard<mutex> lock(mtx9k);
            mRes9k = ml::Yolo9kHandler::process(*mFeed9k);
            mFeed9Updated = false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(16));
    }
}

void Yolo9KTestApp::mouseDown( MouseEvent event )
{
}

void Yolo9KTestApp::update() {
    if(mCapture && mCapture->checkNewFrame() ) {
        mFeed   = mCapture->getSurface();
        mTex    = gl::Texture2d::create(*mFeed);
        if(!mFeed9Updated){
            mFeed9k = Surface8u::create(
                ip::resizeCopy(*mFeed, mFeed->getBounds(),
                ml::Yolo9kHandler::getExpectedSize()));
            mFeed9Updated = true;
        }
    }
}

void Yolo9KTestApp::draw()
{
    gl::clear( Color( 0, 0, 0 ) );
    gl::viewport(toPixels(getWindowSize()));
    gl::setMatricesWindow(toPixels(getWindowSize()));
    
    if(mTex){
        Rectf centeredRect = Rectf( mTex->getBounds() ).getCenteredFit( toPixels(getWindowBounds()), true );
        gl::draw(mTex, centeredRect);
        
        for(auto res : mRes9k){
            Rectf bound;
            bound.x1 = centeredRect.getX1() + res.rect.x1 * centeredRect.getWidth();
            bound.x2 = centeredRect.getX1() + res.rect.x2 * centeredRect.getWidth();
            bound.y1 = centeredRect.getY1() + res.rect.y1 * centeredRect.getHeight();
            bound.y2 = centeredRect.getY1() + res.rect.y2 * centeredRect.getHeight();
                
            auto str = mClass[res.classIndex];
            
            {
                vec2 size = mDisplayTex->measureString(str);
                gl::ScopedColor scpColor(Color::white());
                gl::drawStrokedRect(bound, 4.f);
                gl::drawSolidRect(Rectf(bound.x1 - 2.f, bound.y1-5-size.y, bound.x1 + size.x + 8.f, bound.y1));
            }
            
            {
                gl::ScopedColor scpColor(Color::black());
                mDisplayTex->drawString(str, vec2(bound.x1, bound.y1 - 5));
            }
        }
    }
}

CINDER_APP( Yolo9KTestApp, RendererGl, [](App::Settings *settings){
    settings->setHighDensityDisplayEnabled(false);
    settings->setWindowSize(960, 540);
    settings->setResizable (false);
} )
