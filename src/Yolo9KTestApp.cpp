#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class Yolo9KTestApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void Yolo9KTestApp::setup()
{
}

void Yolo9KTestApp::mouseDown( MouseEvent event )
{
}

void Yolo9KTestApp::update()
{
}

void Yolo9KTestApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) ); 
}

CINDER_APP( Yolo9KTestApp, RendererGl )
