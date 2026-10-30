use eframe::egui::{Pos2, Rect, Response, Vec2};

pub type Point3 = [f32; 3];

const FIELD_OF_VIEW: f32 = 0.9;
const NEAR: f32 = 0.05;
const ORBIT_SPEED: f32 = 0.008;
const ZOOM_SPEED: f32 = 0.0015;
const PITCH_LIMIT: f32 = 1.45;

/// Orbits a target point; z is up, like the simulator's coordinates.
pub struct Camera {
    pub target: Point3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

pub struct Projection {
    eye: Point3,
    right: Point3,
    up: Point3,
    forward: Point3,
    centre: Pos2,
    focal: f32,
}

pub struct Projected {
    pub at: Pos2,
    pub depth: f32,
}

impl Camera {
    pub fn looking_at_room(width: f32, depth: f32, height: f32) -> Self {
        Camera {
            target: [width / 2.0, depth / 2.0, height / 3.0],
            yaw: -2.2,
            pitch: 0.55,
            distance: 1.6 * width.max(depth) + 2.0,
        }
    }

    pub fn follow_mouse(&mut self, response: &Response, scroll: f32) {
        let drag = response.drag_delta();
        self.yaw -= drag.x * ORBIT_SPEED;
        self.pitch = (self.pitch + drag.y * ORBIT_SPEED).clamp(0.05, PITCH_LIMIT);
        if response.hovered() {
            self.distance = (self.distance * (1.0 - scroll * ZOOM_SPEED)).clamp(1.0, 60.0);
        }
    }

    pub fn projection(&self, viewport: Rect) -> Projection {
        let offset = [
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        ];
        let eye = add(self.target, scale(offset, self.distance));
        let forward = normalize(sub(self.target, eye));
        let right = normalize(cross(forward, [0.0, 0.0, 1.0]));
        let up = cross(right, forward);
        let focal = viewport.height() / (2.0 * (FIELD_OF_VIEW / 2.0).tan());
        Projection {
            eye,
            right,
            up,
            forward,
            centre: viewport.center(),
            focal,
        }
    }
}

impl Projection {
    pub fn project(&self, point: Point3) -> Option<Projected> {
        let relative = sub(point, self.eye);
        let depth = dot(relative, self.forward);
        if depth < NEAR {
            return None;
        }
        let x = dot(relative, self.right) / depth;
        let y = dot(relative, self.up) / depth;
        Some(Projected {
            at: self.centre + Vec2::new(x, -y) * self.focal,
            depth,
        })
    }

    pub fn world_to_pixels(&self, world_size: f32, depth: f32) -> f32 {
        world_size * self.focal / depth
    }

    /// How brightly a face with this outward normal is lit, 0.35 (in shadow) to 1.
    pub fn lighting(&self, normal: Point3) -> f32 {
        let light = normalize([-0.4, -0.6, 1.0]);
        0.35 + 0.65 * dot(normalize(normal), light).max(0.0)
    }

    pub fn faces_camera(&self, point_on_face: Point3, normal: Point3) -> bool {
        dot(sub(self.eye, point_on_face), normal) > 0.0
    }
}

pub fn add(a: Point3, b: Point3) -> Point3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: Point3, b: Point3) -> Point3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: Point3, s: f32) -> Point3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn lerp(a: Point3, b: Point3, t: f32) -> Point3 {
    add(a, scale(sub(b, a), t))
}

fn dot(a: Point3, b: Point3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point3, b: Point3) -> Point3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: Point3) -> Point3 {
    let length = dot(a, a).sqrt().max(1e-6);
    scale(a, 1.0 / length)
}
