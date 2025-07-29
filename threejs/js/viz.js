const sphereRadius = 0.05;
const dotDash = 0.002;

// Load JSON data from Python export
const response = await fetch('../viz_data.json');
const data = await response.json();

const { n_layers, embeddings, embeddings_nonsequential, Y, grid, d } = data;

const zStep = d === 3 ? 4.2 : 3.2;

// Build color palette
const classColors = d === 2
  ? [ new THREE.Color(1.0, 0.0, 0.0), new THREE.Color(0.0, 0.0, 1.0) ]  // red, blue
  : [
      new THREE.Color(1.000, 0.729, 0.286), // #ffba49
      new THREE.Color(0.129, 0.643, 0.620), // #20a39e
      new THREE.Color(0.937, 0.357, 0.357)  // #ef5b5b
    ];

// === Scene setup ===
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
//scene.fog = new THREE.Fog(0xffffff, 0, 20);

const frust = d === 3 ? (1 + n_layers) * 2 : (1 + n_layers) * 1.5;
const zMid = (n_layers * zStep) / 2;

const camera = new THREE.OrthographicCamera(-frust, frust, frust, -frust, -100, 100);
if (d === 3) {
  camera.position.set(5, 5, zMid + 1*zStep);
} else {
  camera.position.set(5, 5, zMid + 1*zStep);
}
camera.up.set(0, 0, 1);
camera.lookAt(0, 0, zMid);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
  preserveDrawingBuffer: true
});
renderer.sortObjects = true;
renderer.setClearColor(0xffffff, 0);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(900, 900);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, zMid);
controls.enableDamping = true;
controls.dampingFactor = 0.25;
controls.update();

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const light = new THREE.DirectionalLight(0xffffff, 0.6);
light.position.set(10, 10, 10);
scene.add(light);

// Materials
function createPlaneMaterial(color = 0xf3f3f3, opacity = 0.8, depthWrite = true, renderOrder = 0) {
  const material = new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: opacity,
    depthWrite: depthWrite,
    depthTest: true,
    side: THREE.DoubleSide
  });
  material.renderOrder = renderOrder;
  return material;
}



// Helpers
function makeDashedLine(p1, p2, color, dashSize = 0.01, gapSize = 0.01, width = 2) {
  const points = [new THREE.Vector3(...p1), new THREE.Vector3(...p2)];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  const meshLine = new MeshLine();
  meshLine.setGeometry(geometry);

  const totalSize = dashSize + gapSize;
  const material = new MeshLineMaterial({
    color: new THREE.Color(color),
    lineWidth: width * 0.005, // empirically scaled for visual similarity
    transparent: true,
    dashArray: totalSize,
    dashOffset: 0,
    dashRatio: dashSize / totalSize,
    depthTest: true,
    depthWrite: true,
    fog: true
  });

  return new THREE.Mesh(meshLine, material);
}

function vizMapping2D() {
  // Main visualization for 2D data
  for (let l = 0; l <= n_layers; l++) {
    const z = l * zStep;

    // Grid lines
    [-1, 0, 1].forEach(x => {
      if (x == 0) {
        scene.add(makeDashedLine([x, -1, z], [x, 1, z], 0x777777, 0, 1, 2));
      } else {
        scene.add(makeDashedLine([x, -1, z], [x, 1, z], 0x000000, 0, 1, 2));
      }
    });
    [-1, 0, 1].forEach(y => {
      if (y == 0) {
        scene.add(makeDashedLine([-1, y, z], [1, y, z], 0x777777, 0, 1, 2));
      } else {
        scene.add(makeDashedLine([-1, y, z], [1, y, z], 0x000000, 0, 1, 2));
      }
    });

    // Translucent unit plane for the axes
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), createPlaneMaterial(0xf3f3f3, 0.8, true, 1));
    plane.position.set(0, 0, z);
    scene.add(plane);

    // Wireframe box showing transformed grid (first 4 pts as 2x2)
    if (l < embeddings_nonsequential.length && l > 0) {
      const pts4 = embeddings_nonsequential[l].slice(0, 4);
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          const idx = i * 2 + j;
          const a = pts4[idx];
          if (i < 1) {
            const b = pts4[(i + 1) * 2 + j];
            scene.add(makeDashedLine([a[0], -a[1], z], [b[0], -b[1], z], 0x000000, 0.01, 0.01, 5));
          }
          if (j < 1) {
            const b = pts4[i * 2 + (j + 1)];
            scene.add(makeDashedLine([a[0], -a[1], z], [b[0], -b[1], z], 0x000000, 0.01, 0.01, 5));
          }
        }
      }
    }

    // Data points
    if (Y && Y.length > 0) {
      for (let i = 0; i < Y.length; i++) {
        const [x, y] = embeddings[l][i];
        const color = new THREE.Color(classColors[Y[i] % classColors.length]);
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(sphereRadius, 16, 16),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5 })
        );
        sphere.position.set(x, -y, z);
        scene.add(sphere);
      }
    }

    // Inter-layer lines between data points
    if (l < n_layers && Y && Y.length > 0) {
      for (let i = 0; i < Y.length; i++) {
        const [x0, y0] = embeddings[l][i];
        const [x1, y1] = embeddings[l + 1][i];
        const color = new THREE.Color(classColors[Y[i] % classColors.length]);
        scene.add(makeDashedLine(
          [x0, -y0, z],
          [x1, -y1, z + zStep],
          color,
          dotDash,
          dotDash,
          2.5
        ));
      }
    }
  }

  // Mapping lines from grid
  for (let l = 0; l < n_layers; l++) {
    const z0 = l * zStep;
    const z1 = (l + 1) * zStep;
    for (let i = 0; i < grid.length; i++) {
      const [x0, y0] = grid[i];
      const [x1, y1] = embeddings_nonsequential[l + 1][i];
      scene.add(makeDashedLine(
        [x0, -y0, z0],
        [x1, -y1, z1],
        0x333333,
        0.01,
        0.01,
        5
      ));
    }
  }
}

function vizMapping3D() {
  // Main visualization for 3D data
  for (let l = 0; l <= n_layers; l++) {
    const z = l * zStep;

    // Define corners of the transformed box
    if (l < embeddings_nonsequential.length) {
      const x = embeddings_nonsequential[l].map(pt => pt[0]);
      const y = embeddings_nonsequential[l].map(pt => -pt[1]);
      const z_coords = embeddings_nonsequential[l].map(pt => pt[2] + z);

      // Indices of the 12 edges of a cube (each is a pair of corner indices)
      const edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  // bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  // top face
        [0, 4], [1, 5], [2, 6], [3, 7]   // vertical edges
      ];

      // Plot each edge
      for (const [i, j] of edges) {
        scene.add(makeDashedLine(
          [x[i], y[i], z_coords[i]],
          [x[j], y[j], z_coords[j]],
          0x000000,
          0.01,
          0.01,
          5
        ));
      }
      
      // Create translucent faces for transformed cube
      const transformedFaces = [
        [0, 1, 3, 2],  // bottom
        [4, 5, 7, 6],  // top
        [0, 1, 5, 4],  // front
        [2, 3, 7, 6],  // back
        [0, 2, 6, 4],  // left
        [1, 3, 7, 5]   // right
      ];
      
      for (const face of transformedFaces) {
        const faceGeometry = new THREE.BufferGeometry();
        const faceVertices = [];
        
        // Add all vertices of the face
        for (const vertexIndex of face) {
          faceVertices.push(x[vertexIndex], y[vertexIndex], z_coords[vertexIndex]);
        }
        
        // Create two triangles from the quad (0,1,2 and 0,2,3)
        const indices = [0, 1, 2, 0, 2, 3];
        
        faceGeometry.setAttribute('position', new THREE.Float32BufferAttribute(faceVertices, 3));
        faceGeometry.setIndex(indices);
        faceGeometry.computeVertexNormals();
        
              const faceMesh = new THREE.Mesh(faceGeometry, createPlaneMaterial(0xddddff, 0.3, false, 2));
      scene.add(faceMesh);
      }
    }

    // Plot axes box (unit cube)
    const x = grid.map(pt => pt[0]);
    const y = grid.map(pt => -pt[1]);
    const z_coords = grid.map(pt => pt[2] + z);

    // The 8 corners of the box, ordered consistently
    const verts = x.map((xi, i) => [xi, y[i], z_coords[i]]);

    // Define the 6 faces using the corner indices
    const faces = [
      [0, 1, 3, 2],  // bottom
      [4, 5, 7, 6],  // top
      [0, 1, 5, 4],  // front
      [2, 3, 7, 6],  // back
      [0, 2, 6, 4],  // left
      [1, 3, 7, 5]   // right
    ];

    // Create wireframe and translucent faces for each face
    for (const face of faces) {
      // Create wireframe
      for (let i = 0; i < face.length; i++) {
        const j = (i + 1) % face.length;
        scene.add(makeDashedLine(
          verts[face[i]],
          verts[face[j]],
          0x333333,
          0,
          1,
          5
        ));
      }
      
      // Create translucent face (triangulate the quad)
      const faceGeometry = new THREE.BufferGeometry();
      const faceVertices = [];
      
      // Add all vertices of the face
      for (const vertexIndex of face) {
        faceVertices.push(...verts[vertexIndex]);
      }
      
      // Create two triangles from the quad (0,1,2 and 0,2,3)
      const indices = [0, 1, 2, 0, 2, 3];
      
      faceGeometry.setAttribute('position', new THREE.Float32BufferAttribute(faceVertices, 3));
      faceGeometry.setIndex(indices);
      faceGeometry.computeVertexNormals();
      
              const faceMesh = new THREE.Mesh(faceGeometry, createPlaneMaterial(0xffffff, 0.0, false, 3));
        scene.add(faceMesh);
    }

    // Data points
    if (Y && Y.length > 0) {
      for (let i = 0; i < Y.length; i++) {
        const [x, y, z_coord] = embeddings[l][i];
        const color = new THREE.Color(classColors[Y[i] % classColors.length]);
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(sphereRadius, 16, 16),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.65 })
        );
        sphere.position.set(x, -y, z_coord + z);
        scene.add(sphere);
      }
    }

    // Inter-layer lines between data points
    if (l < n_layers && Y && Y.length > 0) {
      for (let i = 0; i < Y.length; i++) {
        const [x0, y0, z0] = embeddings[l][i];
        const [x1, y1, z1] = embeddings[l + 1][i];
        const color = new THREE.Color(classColors[Y[i] % classColors.length]);
        scene.add(makeDashedLine(
          [x0, -y0, z0 + z],
          [x1, -y1, z1 + z + zStep],
          color,
          dotDash,
          dotDash,
          2.5
        ));
      }
    }
  }

  // Mapping lines from grid
  for (let l = 0; l < n_layers; l++) {
    const z0 = l * zStep;
    const z1 = (l + 1) * zStep;
    for (let i = 0; i < grid.length; i++) {
      const [x0, y0, z0_coord] = grid[i];
      const [x1, y1, z1_coord] = embeddings_nonsequential[l + 1][i];
      scene.add(makeDashedLine(
        [x0, -y0, z0_coord + z0],
        [x1, -y1, z1_coord + z1],
        0x333333,
        0.01,
        0.01,
        5
      ));
    }
  }
}

// Call appropriate function based on dimensionality
if (d === 2) {
  vizMapping2D();
} else if (d === 3) {
  vizMapping3D();
} else {
  console.error('Unsupported dimensionality:', d);
}

// Animate
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();