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
from scipy.optimize import curve_fit
import pylab as py
import time
from OpenGL.GL.shaders import *
import time, sys
program = None
################## constants   ###
l = 1.7; #lacunarity 
p=l**5.0/6.0; #persistence


#################
i0=-12;  #init value of i
i_max=8; # maxmium value of i
#scaling=math.sqrt(l**(1.0/3.0)/(l**(1.0/3.0)-1))# scaling factor of ith layer 

#assert i0<i_max , "i0 has to be bigger than i_max"
# i have set some boundry conditions here, you can remove it simply by uncomment the line below #
#assert i0<=-3, "please put a number <=-3 for i0 to see the complete results." 
#assert i_max>=3, "please put a number >=3 for i_max to see the complete results." 

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

                    //Normalise gradients
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

                    void main(void) {
                            vTexCoord3D = gl_Vertex.xyz * 1.0 + vec3(time/17.0, time/13.0, time);
                            //prime number, less likely to repeat
                            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                            vnoise = snoise(vec3(alpha,alpha,1.0)*vTexCoord3D);
                            
                            
                    }
               
                
            ''',GL_VERTEX_SHADER),
            compileShader(simplexnoise+'''
                    varying vec3 vTexCoord3D;
                    varying float vnoise;
                    uniform float alpha;
                    uniform float persistence;
                    uniform float time;
                    uniform float amp;

                    void main( void )
                    {
                        //single layer   
                        float n = snoise(vec3(alpha,alpha,1.0)*vTexCoord3D);
                        gl_FragColor = vec4(0.5 + 0.5 * n, 0.5+0.5*vnoise, 0, 1.0);
    
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
            
#########        key formula       ##########
        global i
        alpha=l**i
#        print "alpha",alpha
        amp=alpha**(-5.0/6.0)
        
        glUniform1f(timeparam,500*(time.time()-t0))
        glUniform1f(perparam,p) 
        glUniform1f(lacparam,l)
        glUniform1f(alphaparam,alpha)
        glUniform1f(ampparam,amp)
        
        chooseFBO(fbo)
        draw(vertexdata)

        #  since this is double buffered, swap the buffers to display what just got drawn.
        data_r=glReadPixels(0,0,512,512,GL_RED,GL_FLOAT)-0.5
     
        data_g=glReadPixels(0,0,512,512,GL_GREEN,GL_FLOAT)-0.5

        global l_var
        global l_var2
        global l_var3
#        red data- fragment shader
        data=data_r*self.mask
        mymean=np.sum(data)/self.n
        
#        green data - vertex shader
        data=data_g*self.mask
        mymean2=np.sum(data)/self.n
        
#        the difference between green and red data
        data=(data_r-data_g)*self.mask
        mymean3=np.sum(data)/self.n

        #var2 is the old method
        #var2=sum(data*data)/self.n-(sum(data)/self.n)**2 #trying to reduce the round-off error.

        data_a=(data_r-mymean)*self.mask; #this method is better, smaller round off error due to subtraction
        var=sum(data_a*data_a)/self.n
        data_b=(data_g-mymean2)*self.mask;
        var2=sum(data_b*data_b)/self.n
        data=(data_r-data_g-mymean3)*self.mask;
        var3=sum(data*data)/self.n

        l_var.append(var)
        l_var2.append(var2)
        l_var3.append(var3)

        count=count+1;
        if count==200:
                rate=100/(time.time()-mytime)
                mytime=time.time()
                count=0
                print "rate",rate
                data=data_r*self.window2D
                F1=fftpack.fft2(data)
                F2 =fftpack.fftshift(F1)
    
                py.figure(1)
                py.imshow(data)
                py.draw()

                py.figure(2)
                py.imshow(log(F2.real*F2.real+F2.imag*F2.imag+0.0001))
                py.draw()
                
      
                py.figure(3)
        #                var - cyan ; var2 - blue; var3 - difference pink
                ave_var=sum(l_var)/len(l_var)
                ave_var2=sum(l_var2)/len(l_var2)
                ave_var3=sum(l_var3)/len(l_var3)
                
                global y1
                global y1b
                global y1c
                global x1
                y1=np.append(y1,[log10(amp*math.sqrt(ave_var))])
                y1b=np.append(y1b,[log10(amp*math.sqrt(ave_var2))])
                y1c=np.append(y1c,[log10(math.sqrt(ave_var3))])
                
                x1=np.append(x1,[log10(alpha)])
        #                plt.scatter(x1,y1c,c='pink');
                plt.scatter(x1,y1b,c='blue');
                plt.scatter(x1,y1,c='cyan');
                py.draw()
               

############    High pass filter model ##########
                global x2    
                x2=np.append(x2,[log10(alpha)])
                
                global y2
                global y2b
                y2=np.append(y2,[ave_var])
                y2b=np.append(y2b,[ave_var3])
                
                global y3
                y3=np.append(y3,[ave_var*amp*amp])
                
                global x3
                x3=np.append(x3,[i])

                py.figure(5)
#                 log10(alpha) - S^2
                plt.scatter(x2,y2,c='cyan') # fragment shader
                
#                if i-i0>2:
 
                py.draw()

                py.figure(6)
                py.clf()
                if i-i0>15:
                    plt.scatter(x3,y3,c='red');
                    popt,pcov=curve_fit(fitting,x3,y3)
                    print popt
                    print sqrt(pcov)
                    ls=arange(-12,8,0.1)
                    py.plot(ls,fitting(np.array(ls),popt[0]))
                    py.draw()
                    plt.xlabel('scale i')
                    plt.ylabel('Variance*amp*amp')
                
  
                print i
                i=i+1.0
                
                
                if i>i_max:
                    pause=True
                l_var=[]
                l_var2=[]
                l_var3=[]
                
                
                
        chooseFBO(0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw(vertexdata)
        glutSwapBuffers()

def fitting(i,T):
    C=0.193
    return (C*T)**2*l**(i/3.0)/(1+T**2*(l**(2*i)))
   
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
    
 ###########################   
    global y1
    y1=np.array([])
    global y2
    y2=np.array([])
    global y3
    y3=np.array([])
    global y4
    y4=np.array([])
    
    global x1
    x1 =np.array([])
    global x2
    x2 =np.array([])
    global x3
    x3 =np.array([])
    global x4
    x4 =np.array([])
    
    global y1b
    y1b=np.array([])
    global y1c
    y1c=np.array([])
    global y2b
    y2b=np.array([])
    global y2c
    y2c=np.array([])

    
    
    global l_var
    l_var=[]
    global l_var2
    l_var2=[]
    global l_var3
    l_var3=[]
    global power
    power=0
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
#    py.figure(1)
#    py.figure(2)
#    py.figure(3)
#    u = np.arange(-3,4,1.0)
#    plt.plot(u,u*(-5.0/6)-0.75,c='g')
#    plt.plot(u,u*(1.0/6)-0.5,c='g')
#    plt.xlabel('log10(alpha) ~ i')
#    plt.ylabel('log10(Amplitude*Srms)')
#    py.figure(5)
#    u=np.arange(-5,3,0.1)
#    T=1.50
#    C=0.193
#    v=(C*T*10**u)**2/(1+T**2*10**(2*u))
#    plt.plot(u,v)
#    plt.xlabel('log10(alpha)')
#    plt.ylabel('Variance')
    py.figure(6)
    
    # Start Event Processing Engine
    #glutMainLoop()
    py.show()



if __name__ == "__main__":
    main()


