use std::path::Path;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use rayon::prelude::*;
use nalgebra::{Vector3, Matrix3, Point3};
use image::{ImageBuffer, Luma, DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraParams {
    pub position: Vector3<f32>,
    pub rotation: Matrix3<f32>, // rotation matrix
    pub fov_radians: f32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct VoxelGrid {
    pub data: Vec<f32>,
    pub size: usize, // NxNxN cube
    pub voxel_size: f32,
    pub center: Vector3<f32>,
}

#[derive(Debug, Clone)]
pub struct MotionMask {
    pub data: Vec<bool>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<f32>, // grayscale normalized 0-1
    pub width: u32,
    pub height: u32,
}

impl VoxelGrid {
    pub fn new(size: usize, voxel_size: f32, center: Vector3<f32>) -> Self {
        VoxelGrid {
            data: vec![0.0; size * size * size],
            size,
            voxel_size,
            center,
        }
    }

    pub fn get_index(&self, x: usize, y: usize, z: usize) -> usize {
        x * self.size * self.size + y * self.size + z
    }

    pub fn world_to_voxel(&self, world_pos: &Vector3<f32>) -> Option<(usize, usize, usize)> {
        let half_extent = (self.size as f32 * self.voxel_size) / 2.0;
        let local_pos = world_pos - self.center;
        
        let x = (local_pos.x + half_extent) / self.voxel_size;
        let y = (local_pos.y + half_extent) / self.voxel_size;
        let z = (local_pos.z + half_extent) / self.voxel_size;
        
        if x >= 0.0 && x < self.size as f32 && 
           y >= 0.0 && y < self.size as f32 && 
           z >= 0.0 && z < self.size as f32 {
            Some((x as usize, y as usize, z as usize))
        } else {
            None
        }
    }

    pub fn add_value(&mut self, x: usize, y: usize, z: usize, value: f32) {
        let idx = self.get_index(x, y, z);
        self.data[idx] += value;
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header: size, voxel_size
        writer.write_all(&(self.size as u32).to_le_bytes())?;
        writer.write_all(&self.voxel_size.to_le_bytes())?;
        
        // Write voxel data
        for &value in &self.data {
            writer.write_all(&value.to_le_bytes())?;
        }
        
        writer.flush()
    }
}

impl Frame {
    pub fn from_image(img: &DynamicImage) -> Self {
        let gray_img = img.to_luma8();
        let (width, height) = gray_img.dimensions();
        
        let data: Vec<f32> = gray_img
            .pixels()
            .map(|p| p.0[0] as f32 / 255.0)
            .collect();
        
        Frame { data, width, height }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize]
        } else {
            0.0
        }
    }
}

impl MotionMask {
    pub fn from_frames(prev: &Frame, curr: &Frame, threshold: f32) -> Self {
        assert_eq!(prev.width, curr.width);
        assert_eq!(prev.height, curr.height);
        
        let data: Vec<bool> = prev.data
            .iter()
            .zip(curr.data.iter())
            .map(|(p, c)| (p - c).abs() > threshold)
            .collect();
        
        MotionMask {
            data,
            width: prev.width,
            height: prev.height,
        }
    }

    pub fn has_motion(&self, x: u32, y: u32) -> bool {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize]
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct RayStep {
    pub voxel_coords: (usize, usize, usize),
    pub distance: f32,
}

pub fn cast_ray_dda(
    origin: &Vector3<f32>,
    direction: &Vector3<f32>,
    voxel_grid: &VoxelGrid,
    max_distance: f32,
) -> Vec<RayStep> {
    let mut steps = Vec::new();
    
    // Grid bounds
    let half_extent = (voxel_grid.size as f32 * voxel_grid.voxel_size) / 2.0;
    let grid_min = voxel_grid.center - Vector3::new(half_extent, half_extent, half_extent);
    let grid_max = voxel_grid.center + Vector3::new(half_extent, half_extent, half_extent);
    
    // Ray-box intersection
    let mut t_min = 0.0f32;
    let mut t_max = max_distance;
    
    for i in 0..3 {
        let inv_dir = 1.0 / direction[i];
        let t1 = (grid_min[i] - origin[i]) * inv_dir;
        let t2 = (grid_max[i] - origin[i]) * inv_dir;
        
        let t_near = t1.min(t2);
        let t_far = t1.max(t2);
        
        t_min = t_min.max(t_near);
        t_max = t_max.min(t_far);
        
        if t_min > t_max {
            return steps; // No intersection
        }
    }
    
    if t_min < 0.0 {
        t_min = 0.0;
    }
    
    // Start point
    let start_point = origin + direction * t_min;
    
    // Convert to voxel coordinates
    let local_start = start_point - grid_min;
    let mut vx = (local_start.x / voxel_grid.voxel_size).floor() as i32;
    let mut vy = (local_start.y / voxel_grid.voxel_size).floor() as i32;
    let mut vz = (local_start.z / voxel_grid.voxel_size).floor() as i32;
    
    // Step directions
    let step_x = if direction.x > 0.0 { 1 } else { -1 };
    let step_y = if direction.y > 0.0 { 1 } else { -1 };
    let step_z = if direction.z > 0.0 { 1 } else { -1 };
    
    // Calculate delta distances
    let delta_dist_x = (1.0 / direction.x).abs();
    let delta_dist_y = (1.0 / direction.y).abs();
    let delta_dist_z = (1.0 / direction.z).abs();
    
    // Calculate step and initial side_dist
    let mut side_dist_x = if direction.x < 0.0 {
        (local_start.x / voxel_grid.voxel_size - vx as f32) * delta_dist_x
    } else {
        (vx as f32 + 1.0 - local_start.x / voxel_grid.voxel_size) * delta_dist_x
    };
    
    let mut side_dist_y = if direction.y < 0.0 {
        (local_start.y / voxel_grid.voxel_size - vy as f32) * delta_dist_y
    } else {
        (vy as f32 + 1.0 - local_start.y / voxel_grid.voxel_size) * delta_dist_y
    };
    
    let mut side_dist_z = if direction.z < 0.0 {
        (local_start.z / voxel_grid.voxel_size - vz as f32) * delta_dist_z
    } else {
        (vz as f32 + 1.0 - local_start.z / voxel_grid.voxel_size) * delta_dist_z
    };
    
    // Perform DDA
    let mut current_distance = t_min;
    
    while current_distance <= t_max {
        // Check bounds
        if vx >= 0 && vx < voxel_grid.size as i32 && 
           vy >= 0 && vy < voxel_grid.size as i32 && 
           vz >= 0 && vz < voxel_grid.size as i32 {
            steps.push(RayStep {
                voxel_coords: (vx as usize, vy as usize, vz as usize),
                distance: current_distance,
            });
        }
        
        // Find which direction to step
        if side_dist_x < side_dist_y && side_dist_x < side_dist_z {
            side_dist_x += delta_dist_x;
            vx += step_x;
            current_distance = side_dist_x - delta_dist_x + t_min;
        } else if side_dist_y < side_dist_z {
            side_dist_y += delta_dist_y;
            vy += step_y;
            current_distance = side_dist_y - delta_dist_y + t_min;
        } else {
            side_dist_z += delta_dist_z;
            vz += step_z;
            current_distance = side_dist_z - delta_dist_z + t_min;
        }
        
        // Check bounds after stepping
        if vx < 0 || vx >= voxel_grid.size as i32 ||
           vy < 0 || vy >= voxel_grid.size as i32 ||
           vz < 0 || vz >= voxel_grid.size as i32 {
            break;
        }
    }
    
    steps
}

pub fn pixel_to_ray_direction(
    pixel_x: u32,
    pixel_y: u32,
    camera: &CameraParams,
) -> Vector3<f32> {
    // Convert pixel coordinates to normalized device coordinates
    let focal_length = (camera.width as f32 / 2.0) / (camera.fov_radians / 2.0).tan();
    
    let x = (pixel_x as f32 - camera.width as f32 / 2.0) / focal_length;
    let y = -(pixel_y as f32 - camera.height as f32 / 2.0) / focal_length; // Flip Y
    let z = -1.0; // Looking down negative Z in camera space
    
    let camera_ray = Vector3::new(x, y, z).normalize();
    
    // Transform to world space
    camera.rotation * camera_ray
}

pub fn process_frame_motion(
    motion_mask: &MotionMask,
    curr_frame: &Frame,
    camera: &CameraParams,
    voxel_grid: &mut VoxelGrid,
    max_ray_distance: f32,
) {
    // Process motion pixels in parallel
    let motion_pixels: Vec<(u32, u32, f32)> = (0..motion_mask.height)
        .into_par_iter()
        .flat_map(|y| {
            (0..motion_mask.width)
                .into_par_iter()
                .filter_map(move |x| {
                    if motion_mask.has_motion(x, y) {
                        let intensity = curr_frame.get_pixel(x, y);
                        if intensity > 0.01 { // Skip very dark pixels
                            Some((x, y, intensity))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
        })
        .collect();

    // Cast rays and accumulate in voxel grid
    for (pixel_x, pixel_y, intensity) in motion_pixels {
        let ray_direction = pixel_to_ray_direction(pixel_x, pixel_y, camera);
        let ray_steps = cast_ray_dda(
            &camera.position,
            &ray_direction,
            voxel_grid,
            max_ray_distance,
        );
        
        // Accumulate brightness along ray path
        for step in ray_steps {
            let (vx, vy, vz) = step.voxel_coords;
            let distance_attenuation = 1.0 / (1.0 + 0.0001 * step.distance);
            let value = intensity * distance_attenuation * 0.1; // Scale factor
            
            // Thread-safe accumulation (we'll need to handle this differently for true parallelism)
            let idx = voxel_grid.get_index(vx, vy, vz);
            voxel_grid.data[idx] += value;
        }
    }
}

pub fn create_rotation_matrix(yaw: f32, pitch: f32, roll: f32) -> Matrix3<f32> {
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cr = roll.cos();
    let sr = roll.sin();
    
    Matrix3::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_grid_creation() {
        let grid = VoxelGrid::new(100, 0.1, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(grid.data.len(), 100 * 100 * 100);
        assert_eq!(grid.size, 100);
        assert_eq!(grid.voxel_size, 0.1);
    }

    #[test]
    fn test_motion_detection() {
        let frame1_data = vec![0.5; 100];
        let frame2_data = vec![0.7; 100];
        
        let frame1 = Frame { data: frame1_data, width: 10, height: 10 };
        let frame2 = Frame { data: frame2_data, width: 10, height: 10 };
        
        let motion = MotionMask::from_frames(&frame1, &frame2, 0.1);
        assert!(motion.has_motion(0, 0)); // Should detect motion with threshold 0.1
    }

    #[test]
    fn test_ray_casting() {
        let grid = VoxelGrid::new(10, 1.0, Vector3::new(0.0, 0.0, 0.0));
        let origin = Vector3::new(-6.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        
        let steps = cast_ray_dda(&origin, &direction, &grid, 15.0);
        assert!(!steps.is_empty());
    }
}

// Example usage and main function
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Rust Multi-Camera Voxel Object Tracker");
    
    // Example setup
    let voxel_size = 0.1; // 10cm voxels
    let grid_size = 200; // 200x200x200 grid
    let grid_center = Vector3::new(0.0, 0.0, 5.0); // 5m in front
    
    let mut voxel_grid = VoxelGrid::new(grid_size, voxel_size, grid_center);
    
    // Example camera setup
    let camera1 = CameraParams {
        position: Vector3::new(-2.0, 0.0, 0.0),
        rotation: create_rotation_matrix(0.0, 0.0, 0.0),
        fov_radians: 60.0_f32.to_radians(),
        width: 640,
        height: 480,
    };
    
    let camera2 = CameraParams {
        position: Vector3::new(2.0, 0.0, 0.0),
        rotation: create_rotation_matrix(0.0, 0.0, 0.0),
        fov_radians: 60.0_f32.to_radians(),
        width: 640,
        height: 480,
    };
    
    println!("Setup complete. Voxel grid: {}x{}x{}, voxel size: {}", 
             grid_size, grid_size, grid_size, voxel_size);
    println!("Camera 1 position: {:?}", camera1.position);
    println!("Camera 2 position: {:?}", camera2.position);
    
    // In a real application, you would:
    // 1. Load video frames from multiple cameras
    // 2. Detect motion between consecutive frames
    // 3. Process each camera's motion through process_frame_motion()
    // 4. Save the final voxel grid
    
    // Example of saving voxel grid
    voxel_grid.save_to_file("output_voxel_grid.bin")?;
    println!("Saved voxel grid to output_voxel_grid.bin");
    
    Ok(())
}