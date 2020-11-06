**Objective**

Exploring the sky with the delight of appreciating universe’s beauty is thrilling and mind-breaking, however, the reality can be sometimes frustrating. Very often, you will see an unstable blurry image through a telescope due to atmospheric air turbulence, a result of temperature fluctuations in the atmosphere and mixing air parcels at various densities and temperatures. Another common example is the “wavy” effect when looking at heat waves on a sun-baked road or watching a bonfire. 



The principle issue with seeing is misleading results. For instance, close double stars might appear as a single star on a telescope, and a single star might appear as a double star on a hazy day.  This leads to very complicated calculations and analysis, and results may still be inaccurate.



The project aims to develop a turbulence model to help in testing and improving the Adaptive Optical systems (AO systems), a system widely used to correct the astronomical seeing problem. The turbulence model follows the well-known Kolmogorov turbulence theory and it was created based on the Perlin noise algorithm. A C-like shading language, OpenGL and Python were used in this project. 



![The Lyot Project](https://lh3.googleusercontent.com/proxy/NODITobqFDk80hK3xu7nfU6otyLomMuwh1VuFLuvfvwty-6w3YjXd6vf96Nw0fOgK6ONpgUUa9I)

 A basic AO system involves a wavefront sensor, a beam splitter, deformable mirrors and a fast real-time computer, shown in figure above. A distorted light beam first interacts with a deformable mirror, then pass through a beam splitter where it gets split to two, one beam is sent to a high-resolution while the other is collected at a wavefront sensor which checks for the wavefront distortion. 

The adaptive mirror is connected with a computer and is designed to cancel out the atmospheric turbulence. This is where this project will be used, with the correct input from the distorted beam, we can build a dynamic atmospheric model and subtract the atmospheric effect from the beam to get a undistorted beam light. When the “clean” beam is measured at the wavefront sensor, the control computer will compute future calculations based on the turbulence model and send small correction order the the adaptive mirror. It is a feedback control loop, where it monitor the turbulence level and auto correct the mirror shape to make sure the image on camera is always clear. 



**How to run**

- ![simplex_update.html](https://placehold.it/15/f03c15/000000?text=+) `simplex_update.html`

Open simplex_update.html and adjust the parameters to see the change.






