import OpenGL
OpenGL.ERROR_ON_COPY = True
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import pylab as py
import time
from OpenGL.GL.shaders import *
import time, sys
program = None
################## constants   ###
l = math.sqrt(math.sqrt(10.0)); #lacunarity 
p=l**(5.0/6.0); #persistence
A=1.61
#################


#amp=(1.0/p)**i 
# an alternative way of expressing it
# alpha is scaling factor
# i is index power
# l is Lacunarity
# p is persistence  p=(l)^(5/6)

def setup_program():
    if not glUseProgram:
            print 'Missing Shader Objects!'
            sys.exit(1)
    global program
    simplexnoise='''
                    // Description : Array and textureless GLSL 2D/3D/4D simplex
                    //               noise functions.
                    //      Author : Ian McEwan, Ashima Arts.
                    //  Maintainer : ijm
                    //     Lastmod : 20110822 (ijm)
                    //     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
                    //               Distributed under the MIT License. See LICENSE file.
                    //               https://github.com/ashima/webgl-noise 
                    // this code is written for 3D, x0,x1,x2,x3. we only using 3 vertex. 
                    vec3 mod289(vec3 x) {
                        return x - floor(x * (1.0 / 289.0)) * 289.0;
                    }

                    vec4 mod289(vec4 x) {
                        return x - floor(x * (1.0 / 289.0)) * 289.0;
                    }

                    vec4 permute(vec4 x) {
                             return mod289(((x*34.0)+1.0)*x);
                    }

                    vec4 taylorInvSqrt(vec4 r)
                    {
                        return 1.79284291400159 - 0.85373472095314 * r;
                    }

                    float snoise(vec3 v)
                        {
                        const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
                        const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

                    // First corner
                        vec3 i  = floor(v + dot(v, C.yyy) ); 
                        // C.yyy gives (1/3,1/3,1/3)
                        vec3 x0 =   v - i + dot(i, C.xxx) ;

                    // Other corners
                        vec3 g = step(x0.yzx, x0.xyz);
                        vec3 l = 1.0 - g;
                        vec3 i1 = min( g.xyz, l.zxy );
                        vec3 i2 = max( g.xyz, l.zxy );

                        //   x0 = x0 - 0.0 + 0.0 * C.xxx;
                        //   x1 = x0 - i1  + 1.0 * C.xxx;
                        //   x2 = x0 - i2  + 2.0 * C.xxx;
                        //   x3 = x0 - 1.0 + 3.0 * C.xxx;
                        vec3 x1 = x0 - i1 + C.xxx;
                        vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
                        vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

                    // Permutations
                        i = mod289(i);
                        vec4 p = permute( permute( permute(
                                             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                                         + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                                         + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

                    // Gradients: 7x7 points over a square, mapped onto an octahedron.
                    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
                        float n_ = 0.142857142857; // 1/7
                        vec3  ns = n_ * D.wyz - D.xzx;

                        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

                        vec4 x_ = floor(j * ns.z);
                        vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

                        vec4 x = x_ *ns.x + ns.yyyy;
                        vec4 y = y_ *ns.x + ns.yyyy;
                        vec4 h = 1.0 - abs(x) - abs(y);

                        vec4 b0 = vec4( x.xy, y.xy );
                        vec4 b1 = vec4( x.zw, y.zw );

                        //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
                        //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
                        vec4 s0 = floor(b0)*2.0 + 1.0;
                        vec4 s1 = floor(b1)*2.0 + 1.0;
                        vec4 sh = -step(h, vec4(0.0));

                        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
                        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

                        vec3 p0 = vec3(a0.xy,h.x);
                        vec3 p1 = vec3(a0.zw,h.y);
                        vec3 p2 = vec3(a1.xy,h.z);
                        vec3 p3 = vec3(a1.zw,h.w);

                    //Normalise gradients, they used taylor expansion to calculate sqrt
                       // vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
                        vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
                        p0 *= norm.x;
                        p1 *= norm.y;
                        p2 *= norm.z;
                        p3 *= norm.w;

                    // Mix final noise value
                        vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
                        return 40.0 * dot( m*m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
                        
                        // explain in thesis
                        //p is direction vec. summation of four colour, RGB, alpha.
                        // feed one corr pt x, find where neighbouring vertices, noise contribution from each vertex, then sum it. 
                        //then noise kernel function. then you get a single floating point number. 
                        }
    '''
    program = compileProgram(
            compileShader(simplexnoise+'''
                    varying vec3 vTexCoord3D;
                    varying float vnoise;
                    uniform float alpha;
                    uniform float time;
                    uniform float lac;
                    uniform float persistence;

                    void main(void) {
                            vTexCoord3D = gl_Vertex.xyz * 1.0 + vec3(time/17.0, time/13.0, time);
                            
                            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                            
                            
                            float amp12=pow(1.0/persistence,-12.0); //amp at i=-12
                            float alpha12=0.001;        //alpha at i=-12
                            // add up layer from -inifinity to -12
                            float l3=pow(lac,1.0/3.0);
                            float s=sqrt(l3/(l3-1.0))*amp12;
                            vnoise =s*snoise(alpha12*vTexCoord3D); 
                            vnoise=vnoise-s*snoise(alpha12*vec3(time/17.0,time/13.0,time));

                            for (int i=-11;i<0;i++){
                                amp12=amp12/persistence;
                                alpha12=alpha12*lac;
                                vnoise=vnoise+amp12*snoise(alpha12*vTexCoord3D);
                                vnoise=vnoise-amp12*snoise(alpha12*vec3(time/17.0,time/13.0,time));
                                
                            }        
                            vnoise=vnoise+snoise(vTexCoord3D);                   

                                     
                             
                                                        
                            
                            
                    }
               
                
            ''',GL_VERTEX_SHADER),
            compileShader(simplexnoise+'''
                    varying vec3 vTexCoord3D;
                    varying float vnoise;
                    uniform float persistence;
                    uniform float time;
                    uniform float lac;
                    uniform int layers;

                    void main( void )
                    {
                        
                        //multi-layer. cos opengl does not support power
                       
                        int layers=8;
                        float n=vnoise;
                        
                        //everything looks fine if uncomment following line
                       // float n=0.0; 
                        float amp=1.0;
                        float alpha_1=1.0;
                     
                        for (int i=1;i<=layers;i++){
                                alpha_1=alpha_1*lac;
                                amp=amp/persistence;
                                n=n+amp*snoise(vec3(alpha_1,alpha_1,1.0)*vTexCoord3D);
                        }
                        gl_FragColor = vec4(0.5 + .501 * vec3(n,n,n), 1.0);
                        }
                    
                    
                    
                 
                         
    ''',GL_FRAGMENT_SHADER))
    global timeparam
    global perparam
    global lacparam
    global alphaparam
    global ampparam
    timeparam = glGetUniformLocation(program,'time')
    perparam = glGetUniformLocation(program,'persistence')
    lacparam = glGetUniformLocation(program,'lac')
    alphaparam = glGetUniformLocation(program,'alpha')
    ampparam = glGetUniformLocation(program,'amp')
    layerparam = glGetUniformLocation(program,'layers')
    print "layers",layerparam
    #    global iparam
    #    iparam = glGetUniformLocation(program,'i')

def setup_vertex_arrays(x0,y0,w,h,n):
    global vertexdata
    np_array_list=[]
    nx=round(w/(n*2.0/sqrt(3.0)))
    m=floor(sqrt(3.0)*n/2.0-0.5)
    dx=w/m
    for ii in arange(0,n):
        x=x0
        dx=w/(m+0.5)
        dy=h/n
        y=y0+ii*dy
        if ii%2 == 1:
            x=x0+w
            dx=-dx
        np_array_list.append([x,y])
        np_array_list.append([x,y+dy])
        np_array_list.append([x+dx/2,y])

        for j in arange(0,m):
            np_array_list.append([x+dx,y+dy])
            np_array_list.append([x+3*dx/2,y])
            x=x+dx
        np_array_list.append([x+dx/2,y+dy])
    vertexdata=np.asarray(np_array_list,dtype=np.float32)
    return vertexdata
#    get ready for VBO


def createFBO(w,h):
    glEnable(GL_TEXTURE_RECTANGLE)
    global texid
    global fbo
    global myimage
    myimage=None
    
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo) 
    
    # can be GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER, or GL_FRAMEBUFFER. The last option sets the framebuffer for both reading and drawing.
    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_RECTANGLE,texid)
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, myimage)
    glTexParameterf(GL_TEXTURE_RECTANGLE,GL_TEXTURE_WRAP_S,GL_CLAMP)
    glTexParameterf(GL_TEXTURE_RECTANGLE,GL_TEXTURE_WRAP_T,GL_CLAMP)
    glTexParameterf(GL_TEXTURE_RECTANGLE,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
    glTexParameterf(GL_TEXTURE_RECTANGLE,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE)
    glBindTexture(GL_TEXTURE_RECTANGLE, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, texid, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

def chooseFBO(fbo):
    #default butter
    if fbo == 0 :
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glReadBuffer(GL_BACK)
        glDrawBuffer(GL_BACK)
    else :
        global texid
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glBindTexture(GL_TEXTURE_RECTANGLE, texid)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glDrawBuffer(GL_COLOR_ATTACHMENT0)


def draw(vertexdata):
#    create and bind VBO / transfer the vertex information to vbo
    vertexbo=vbo.VBO(vertexdata)
    vertexbo.bind()
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2,GL_FLOAT, 0, vertexbo )
    glDrawArrays(GL_TRIANGLE_STRIP,0,len(vertexdata))
    vertexbo.unbind()
    
def phase_power(C,T):
    
    return None

def ReSizeGLScene(Width, Height):
    if Height == 0:                        # Prevent A Divide By Zero If The Window Is Too Small
        Height = 1
    if Width == 0:                        # Prevent A Divide By Zero If The Window Is Too Small
        Width = 1

    glViewport(0, 0, Width, Height)        # Reset The Current Viewport And Perspective Transformation
    glLoadIdentity()
    print Width, Height
    if Width>Height:
        gluOrtho2D(-0.5*Width/Height,0.5*Width/Height,-0.5,0.5)
        print -0.5*Width/Height
    else:
        gluOrtho2D(-0.5,0.5,-0.5*Height/Width,0.5*Height/Width)
        print -0.5*Height/Width

def keyPressed(*args):
    # If escape is pressed, kill everything.
    global i;
    global pause;
    if args[0] == 'q':
        sys.exit()
    if args[0] == 'f':
        glutFullScreen()
    if args[0] == 'b':
        glutReshapeWindow(640,480)
    if args[0] == 'p':
        pause = not(pause)
    if args[0] == 'u':
        i=i+1.0
        print "power index i ",i
        #glUniform1f(scaleparam,0.09)
    if args[0] == 'd':
        i=i-1.0
        print "power index i ",i

class Complex_Noise:

    def __init__(self,Width, Height):                # We call this right after our OpenGL window is created.
        glClearColor(0.0, 0.0, 0.0, 0.0)    # This Will Clear The Background Color To Black
        glClearDepth(1.0)                    # Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)                # Passes if the incoming depth value is less than 1
        glEnable(GL_DEPTH_TEST)                # do depth comparisons and update the depth buffer.
        glShadeModel(GL_SMOOTH)                # Enables Smooth Color Shading
        setup_program()
        global vertexdata
        vertexdata=setup_vertex_arrays(-0.5,-0.5,1.0,1.0,80)

        #windows function
        x=np.arange(-1,1.001,2.0/511)
        y=np.arange(-1,1.001,2.0/511)
        xx,yy=np.meshgrid(x,y)  # gives 256 numbers, in 255 steps. from -1
        r=sqrt(xx**2+yy**2)  #numpy array
        self.window2D=cos(r*math.pi/2)**2*(r<1)
        self.mask=1.0*(r<1)
        self.n=sum(self.mask)
        
    def DrawGLScene(self):
        global pause;
        if (pause):
            time.sleep(0.01)
            return
        if program:
            global mytime
            global count
            global data
            global glutwin
            glUseProgram(program)
               
        glUniform1f(timeparam,100*(time.time()-t0))
        glUniform1f(perparam,p) 
        glUniform1f(lacparam,l)
        
        chooseFBO(fbo)
        draw(vertexdata)

        #  since this is double buffered, swap the buffers to display what just got drawn.
        data_r=A*glReadPixels(0,0,512,512,GL_RED,GL_FLOAT)-0.5
        data_g=A*glReadPixels(0,0,512,512,GL_GREEN,GL_FLOAT)-0.5

       
#        red data- fragment shader
        data=data_r*self.mask
        count=count+1;
        
        if count%200==0:
                print "count",count
                rate=200/(time.time()-mytime)
                mytime=time.time()
#                count=0
                print "rate",rate
                data=data_r*self.window2D
   
    
    
               
                py.figure(1)
                py.imshow(data)
                py.draw()
                
######################################   
                if count%10000==0:
                    pause=True

        chooseFBO(0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw(vertexdata)
        glutSwapBuffers()


def main():
    global pause
    pause = False

    # For now we just pass glutInit one empty argument. I wasn't sure what should or could be passed in (tuple, list, ...)
    # Once I find out the right stuff based on reading the PyOpenGL source, I'll address this.
    glutInit(sys.argv)

    # Select type of Display mode:
    #  Double buffer
    #  RGBA color
    # Alpha components supported
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE )

    # get a 640 x 480 window
    glutInitWindowSize(512, 512)

    # the window starts at the upper left corner of the screen
    glutInitWindowPosition(0, 0)


    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python (like myself), remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main.
    glutCreateWindow("kolmogrov turbulence modelling")


    # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.
    Noise=Complex_Noise(512,512)
    createFBO(512,512)
    
    global vertexattributelocation
    global mytime
    global t0
    mytime = time.time()
    t0 = mytime
    global count
    count=0
    global i
    i=i0
    global layerno
    layerno=2
 ###########################   
    global l_i
    l_i=[]
    global sum_F3
    sum_F3=np.arange(255)*0;
    global l_var
    l_var=[]

    
    global layers
    layers=[]
    global F5
    F5=np.zeros((512,512))
   

    glutDisplayFunc(Noise.DrawGLScene)

    glutIdleFunc(Noise.DrawGLScene)
    # Register the function called when our window is resized.
    glutReshapeFunc(ReSizeGLScene)
    glutKeyboardFunc(keyPressed)

    # Register the function called when the keyboard is pressed.
#	glutKeyboardFunc(keyPressed)
    # When we are doing nothing, redraw the scene.

    print "Information",glGetString(GL_VERSION)
    print glGetString(GL_EXTENSIONS);
    # open whatever python figure windows you want
    py.figure(1)

    # Start Event Processing Engine
    #glutMainLoop()
    py.show()



if __name__ == "__main__":
    main()
 

