# Procedural Grass Renderer in Vulkan
This project showcases an implementation of grass rendering using both a standard vertex pipeline and a modern mesh pipeline using the Vulkan graphics API.

# Controls
By default when the program starts it will take mouse control to turn the camera.

**WASD** - standard first person camera movement  
**Left Shift** - Speeds up the camera while held down  
**E/Q** - to go up and down  
**ESC** - toggles mouse control  

Numbers 1 through 6 can be used to snap the camera to the positions used for gathering performance data.

# Exposed parameters
Several parameters for rendering can be tweaked in real time through the user interface, these are seperated into 3 sections. By default they have values which I believe make for a nice grass simulation.

## Grass parameters
**Grass Lean** - Controls how strong the lean of the grass is, from straight up to fully leaning over.  
**Grass Height** - Controls the height of the blades.  
**Grass Thickness** - Controls how thick the blades are.  
**Curve** - Directly controls the control point of the grass blades Bezier curve, allowing the blades to go from straight to very strongly curved.  

## Wind parameters
**Wind Strength** - Controls how strong the wind is. Higher values will blow the grass over more.  
**Wind Speed** - Controls how fast the wind moves.  
**Wind direction** - Controls the direction the wind is traveling in, from 0 to 2pi.  

## Rendering options
**Use Mesh shaders?** - This checkbox allows users to swap between the vertex shader pipeline and the mesh shader pipeline for rendering.  
**Culling radius** - This parameter controls the radius for the frustum culling operation, values lower than 0 will cull grass within view at the edges of the screen, values higher than 0 will add padding outside the screen to prevent the tips of blades from popping in.  
**LoD dist** - For the Vertex pipeline this value controls the distance from the camera at which a blade will swap to using the low LoD model. For the mesh pipeline it is the distance from the camera where the blade will reach the lowest amount of vertices.  
**Grass col** - This dropdown list allows you to pick from a variety of options to showcase different things using the colour of the grass blade:
 - **Lit** - The standard option, showcases a simple lighting model to showcase a more realistic appearance for the grass.
 - **Unlit** - Removes lighting from the grass, simply colouring it according to the grass texture.
 - **LoD** - Allows users to visualise the blades LoDs through the colour of the blade, where a white grass blade is the highest level of detail and darker colours showcase increasingly lower levels of detail.
 - **Clump** - To highlight the differences in rendering approaches, this option when on the vertex shader pipeline will colour each blade differently, and when on the mesh shader pipeline will colour each clump a unique colour. This allows users to easily visualise the differences to how geometry is processed.
 - **Wireframe** - This option enables a wireframe view allowing users to see blade geometry and how it can change in real-time.

