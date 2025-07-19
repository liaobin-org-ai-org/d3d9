import pyglet
from pyglet import gl, graphics
from pyglet.window import Window
import os
from PIL import Image
import numpy as np
import math

# Global window and resource management
g_window = None
g_title = "D3D9 Window"
g_file_path = "."
g_icon = None
g_models = {}  # Cached loaded models
g_textures = {}  # Cached loaded textures


class D3D9Window(Window):
    """D3D9-style window class for 3D scene rendering"""
    def __init__(self, title=None, file_path=None, icon_path=None):
        super().__init__(width=800, height=600, caption=title, resizable=True)
        
        # Set up Chinese font support
        try:
            pyglet.font.add_file('simhei.ttf')
            print("Successfully loaded Chinese font")
        except Exception as e:
            print(f"Warning: Failed to load Chinese font - {e}")
            print("Hint: Please place the simhei.ttf font file in the project directory")
        
        # Initialize resource path
        global g_file_path, g_icon
        g_file_path = file_path if file_path else "."
        if not os.path.exists(g_file_path):
            os.makedirs(g_file_path)
        
        # Set window icon
        if icon_path and os.path.exists(icon_path):
            try:
                img = pyglet.image.load(icon_path)
                self.set_icon(img)
                g_icon = img
            except Exception as e:
                print(f"Warning: Failed to load window icon - {e}")
        
        self.models = []  # Store models to be rendered
        self.frame_count = 0  # Frame counter
        self.gl_initialized = False  # OpenGL state initialization flag

    def on_draw(self):
        # Initialize OpenGL state on first draw
        if not self.gl_initialized:
            self.initialize_opengl()
            self.gl_initialized = True
            
        self.clear()
        self.setup_3d_view()  # Set up 3D perspective
        self.frame_count += 1  # Update frame count
        # Render all models
        for model in self.models:
            model.update(self.frame_count)
            model.render()

    def initialize_opengl(self):
        """Initialize OpenGL state"""
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)  # Background color
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing (3D occlusion)
        gl.glEnable(gl.GL_TEXTURE_2D)  # Enable textures
        print("OpenGL state initialized")

    def setup_3d_view(self):
        """Set up perspective projection and camera position"""
        width, height = self.get_size()
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(60, width / height, 0.1, 1000)  # Perspective parameters
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.gluLookAt(5, 5, 5,  # Camera position
                     0, 0, 0,  # Target point
                     0, 1, 0)  # Up direction

    def on_resize(self, width, height):
        """Reset view when window size changes"""
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(60, width / height, 0.1, 1000)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        return super().on_resize(width, height)


class BaseModel:
    """Base class for all 3D models"""
    def __init__(self, texture_path=None, color=None):
        self.position = [0, 0, 0]  # Position [x, y, z]
        self.rotation = [0, 0, 0]  # Rotation [x, y, z] (degrees)
        self.scale = [1, 1, 1]     # Scale [x, y, z]
        self.texture = None
        self.vertex_list = None
        self.children = []         # Child models
        # New: Color property, default to white
        self.color = color or (1.0, 1.0, 1.0, 1.0)  # RGBA format
        
        if texture_path:
            success = self.set_texture(texture_path)
            if not success and not color:
                # Use default color if texture loading fails and no color specified
                self.color = (1.0, 0.5, 0.0, 1.0)  # Orange as default color
    
    def set_texture(self, texture_path):
        """Set model texture, ensuring a bindable Texture object is obtained"""
        global g_textures
        
        # Check texture cache
        if texture_path in g_textures:
            cached_tex = g_textures[texture_path]
            self.texture = self._get_bindable_texture(cached_tex)
            return True
            
        if not os.path.exists(texture_path):
            print(f"Warning: Texture file does not exist - {texture_path}")
            return False
        
        # Actually load the texture file
        try:
            # Load texture using pyglet
            tex = pyglet.image.load(texture_path).get_texture()
            g_textures[texture_path] = tex
            self.texture = self._get_bindable_texture(tex)
            print(f"Successfully loaded texture: {texture_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load texture - {e}")
            return False
    
    def _get_bindable_texture(self, texture):
        """Ensure a bindable Texture object is obtained, handling all possible nesting"""
        if texture is None:
            return None
        
        # Recursively find bindable texture object
        current = texture
        while True:
            # Check for bind method
            if hasattr(current, 'bind') and callable(current.bind):
                return current
            # Check for texture property
            if hasattr(current, 'texture'):
                current = current.texture
            else:
                # No bindable texture found, return None
                return None
    
    def update(self, frame_count):
        """Update model state (called every frame), implementing rotation animation"""
        # Make model rotate automatically (increase by 1 degree per frame)
        self.rotation[0] = frame_count * 0.5  # X-axis rotation
        self.rotation[1] = frame_count * 0.8  # Y-axis rotation
        for child in self.children:
            child.update(frame_count)
    
    def render(self):
        """Render the model"""
        gl.glPushMatrix()  # Save current matrix state
        
        # Apply model transformations
        gl.glTranslatef(*self.position)
        gl.glRotatef(self.rotation[0], 1, 0, 0)  # X-axis rotation
        gl.glRotatef(self.rotation[1], 0, 1, 0)  # Y-axis rotation
        gl.glRotatef(self.rotation[2], 0, 0, 1)  # Z-axis rotation
        gl.glScalef(*self.scale)
        
        # Bind texture if available
        if self.texture:
            gl.glEnable(gl.GL_TEXTURE_2D)
            tex = self._get_bindable_texture(self.texture)
            if tex:
                tex.bind()
        else:
            gl.glDisable(gl.GL_TEXTURE_2D)
            # Use color when no texture
            gl.glColor4f(*self.color)
        
        # Draw the model
        if self.vertex_list:
            self.vertex_list.draw(gl.GL_TRIANGLES)
        
        # Restore default color
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        
        # Render child models
        for child in self.children:
            child.render()
        
        gl.glPopMatrix()  # Restore matrix state


class CubeModel(BaseModel):
    """Cube model"""
    def __init__(self, x=1, y=1, z=1, texture_path=None, color=None):
        super().__init__(texture_path, color)
        self.x, self.y, self.z = x, y, z
        self.vertex_list = self._create_vertices()

    def _create_vertices(self):
        """Create vertex data for the cube"""
        # 8 vertex coordinates of the cube (x, y, z)
        vertices = [
            # Front face
            (-self.x/2, -self.y/2, self.z/2),
            (self.x/2, -self.y/2, self.z/2),
            (self.x/2, self.y/2, self.z/2),
            (-self.x/2, self.y/2, self.z/2),
            # Back face
            (-self.x/2, -self.y/2, -self.z/2),
            (self.x/2, -self.y/2, -self.z/2),
            (self.x/2, self.y/2, -self.z/2),
            (-self.x/2, self.y/2, -self.z/2),
        ]
        # Texture coordinates (0-1 range)
        tex_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # 6 faces, each consisting of 2 triangles (36 vertex indices total)
        indices = [
            0, 1, 2, 0, 2, 3,  # Front face
            4, 5, 6, 4, 6, 7,  # Back face
            0, 4, 7, 0, 7, 3,  # Left face
            1, 5, 6, 1, 6, 2,  # Right face
            3, 2, 6, 3, 6, 7,  # Top face
            0, 1, 5, 0, 5, 4   # Bottom face
        ]   
    
        # Collect position and texture coordinate data
        positions = []
        texs = []
        for i in indices:
            x, y, z = vertices[i]
            u, v = tex_coords[i % 4]  # Reuse texture coordinates cyclically
            positions.extend([x, y, z])
            texs.extend([u, v])
    
        # Create pyglet vertex list
        return graphics.vertex_list_indexed(
            len(indices),  # Number of vertices
            list(range(len(indices))),  # Sequential indices
            ('v3f/static', positions),  # Position data
            ('t2f/static', texs)        # Texture coordinate data
        )


class OBJModel(BaseModel):
    """OBJ model loader"""
    def __init__(self, obj_path, texture_path=None, color=None):
        super().__init__(texture_path, color)
        self.vertices = []  # Vertex coordinates (x,y,z)
        self.tex_coords = []  # Texture coordinates (u,v)
        self.indices = []  # Vertex indices
        self._load_obj(obj_path)
        self.vertex_list = self._create_vertex_list()

    def _load_obj(self, obj_path):
        """Load OBJ file and parse vertices, texture coordinates, and indices"""
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
            
        print(f"Loading OBJ model: {obj_path}")
            
        with open(obj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':  # Vertex coordinates
                    self.vertices.append(tuple(map(float, parts[1:4])))
                elif parts[0] == 'vt':  # Texture coordinates
                    self.tex_coords.append(tuple(map(float, parts[1:3])))
                elif parts[0] == 'f':  # Face data (parse as triangles)
                    face = []
                    for p in parts[1:]:
                        # Parse v/vt/vn format, only take vertex and texture coordinate indices
                        v_idx, vt_idx, _ = (int(x)-1 if x else -1 for x in p.split('/'))
                        face.append((v_idx, vt_idx))
                    # Split polygon into triangles
                    for i in range(1, len(face)-1):
                        self.indices.extend([face[0], face[i], face[i+1]])
    
    def _create_vertex_list(self):
        """Create vertex list for rendering"""
        positions = []
        texs = []
        index_data = []
        
        for i, (v_idx, vt_idx) in enumerate(self.indices):
            x, y, z = self.vertices[v_idx]
            u, v = self.tex_coords[vt_idx] if vt_idx != -1 else (0, 0)
            positions.extend([x, y, z])
            texs.extend([u, v])
            index_data.append(i)
        
        # Create vertex list
        return graphics.vertex_list_indexed(
            len(self.indices), index_data,
            ('v3f/static', positions),
            ('t2f/static', texs)
        )


class SphereModel(BaseModel):
    """Sphere model"""
    def __init__(self, radius=1, texture_path=None, slices=20, stacks=20, color=None):
        super().__init__(texture_path, color)
        self.radius = radius
        self.slices = slices
        self.stacks = stacks
        self.vertex_list = self._create_vertices()

    def _create_vertices(self):
        positions = []
        texs = []
        indices = []

        for i in range(self.stacks + 1):
            stack_angle = math.pi * i / self.stacks
            for j in range(self.slices + 1):
                slice_angle = 2 * math.pi * j / self.slices

                x = self.radius * math.cos(slice_angle) * math.sin(stack_angle)
                y = self.radius * math.cos(stack_angle)
                z = self.radius * math.sin(slice_angle) * math.sin(stack_angle)

                u = j / self.slices
                v = i / self.stacks

                positions.extend([x, y, z])
                texs.extend([u, v])

        for i in range(self.stacks):
            for j in range(self.slices):
                first = i * (self.slices + 1) + j
                second = first + self.slices + 1

                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        return graphics.vertex_list_indexed(
            len(positions) // 3, indices,
            ('v3f/static', positions),
            ('t2f/static', texs)
        )


def d3d9main(title=None, file=None, icon=None):
    """Create D3D9-style window"""
    global g_window, g_title
    g_title = title or "D3D9 Window"
    g_window = D3D9Window(g_title, file or ".", icon)
    print(f"D3D9 window created successfully: {g_title}")
    return g_window


def d3d9cube(x=1.0, y=1.0, z=1.0, texture_path=None, color=None):
    """Create cube model"""
    if not g_window:
        raise Exception("Please call d3d9main() first to initialize the window")
    
    cube = CubeModel(x, y, z, texture_path, color)
    g_window.models.append(cube)
    return cube


def load_obj_model(obj_path, texture_path=None, color=None):
    """Load OBJ model"""
    if not g_window:
        raise Exception("Please call d3d9main() first to initialize the window")
    
    global g_models
    
    # Check model cache
    if obj_path in g_models:
        print(f"Using cached OBJ model: {obj_path}")
        model = g_models[obj_path]
        # Update texture if new one provided
        if texture_path:
            model.set_texture(texture_path)
        # Update color if new one provided
        if color:
            model.color = color
    else:
        model = OBJModel(obj_path, texture_path, color)
        g_models[obj_path] = model
    
    g_window.models.append(model)
    print(f"Loaded OBJ model: {obj_path}")
    return model


def d3d9sphere(radius=1, texture_path=None, slices=20, stacks=20, color=None):
    """Create sphere model"""
    if not g_window:
        raise Exception("Please call d3d9main() first to initialize the window")
    
    sphere = SphereModel(radius, texture_path, slices, stacks, color)
    g_window.models.append(sphere)
    return sphere


def d3psd(image_path, output_path=None):
    """
    Convert image to PSD format (fully compatible with psd-tools 1.10.8 lowest-level API)
    """
    import os
    from PIL import Image
    import numpy as np
    import psd_tools
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    
    # Check if input file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Handle output path
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}.psd"
    else:
        output_path = os.path.splitext(output_path)[0] + ".psd"
    
    try:
        # Open and process image
        with Image.open(image_path) as img:
            # Handle transparent channels (convert to white background)
            if img.mode in ('RGBA', 'LA'):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            
            # Create PSD file
            psd = PSDImage.new('RGB', (width, height))
            
            # Convert PIL image to numpy array and adjust to BGR format
            img_np = np.array(img, dtype=np.uint8)
            img_bgr = img_np[..., ::-1]  # RGB to BGR
            
            # Prepare channel data
            channels = [
                img_bgr[..., 0].tobytes(),  # Blue channel
                img_bgr[..., 1].tobytes(),  # Green channel
                img_bgr[..., 2].tobytes()   # Red channel
            ]
            
            # Create layer record (using minimal necessary parameters)
            layer_record = {
                'bbox': (0, 0, width, height),
                'name': "Layer 1",
                'blend_mode': 'normal',  # Use string value directly
                'opacity': 255,
                'visible': True
            }
            
            # Create pixel layer
            layer = PixelLayer(
                psd=psd,
                record=layer_record,
                channels=channels,
                parent=None
            )
            
            # Add layer to PSD (using most basic method)
            psd._layers = [layer]  # Set layer list directly
            
            # Save PSD file
            psd.save(output_path)
        
        print(f"Successfully converted to PSD format: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Failed to convert to PSD: {str(e)}")
        import traceback
        traceback.print_exc()
        return image_path


def set_model_texture(model, texture_path):
    """Set texture for model"""
    if not os.path.exists(texture_path):
        raise FileNotFoundError(f"Texture file not found: {texture_path}")
    
    model.set_texture(texture_path)
    print(f"Applied texture to model: {texture_path}")
    return model.texture


def set_model_color(model, color):
    """Set color for model"""
    model.color = color
    print(f"Set model color: {color}")
    return model.color


def run():
    """Start rendering loop"""
    if g_window:
        print("Starting rendering loop...")
        pyglet.app.run()
    else:
        raise Exception("Please call d3d9main() first to initialize the window")