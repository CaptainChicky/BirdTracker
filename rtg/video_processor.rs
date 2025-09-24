// video_processor.rs
// Example module showing how to integrate video processing with the voxel tracker

use crate::{VoxelGrid, CameraParams, Frame, MotionMask, process_frame_motion, create_rotation_matrix};
use nalgebra::Vector3;
use std::path::{Path, PathBuf};
use std::fs;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct VideoConfig {
    pub camera_id: String,
    pub video_path: String,
    pub camera_params: CameraConfigParams,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CameraConfigParams {
    pub position: [f32; 3],
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub roll_degrees: f32,
    pub fov_degrees: f32,
    pub width: u32,
    pub height: u32,
}

impl From<CameraConfigParams> for CameraParams {
    fn from(config: CameraConfigParams) -> Self {
        CameraParams {
            position: Vector3::new(config.position[0], config.position[1], config.position[2]),
            rotation: create_rotation_matrix(
                config.yaw_degrees.to_radians(),
                config.pitch_degrees.to_radians(),
                config.roll_degrees.to_radians(),
            ),
            fov_radians: config.fov_degrees.to_radians(),
            width: config.width,
            height: config.height,
        }
    }
}

pub struct VideoProcessor {
    pub configs: Vec<VideoConfig>,
    pub motion_threshold: f32,
    pub max_ray_distance: f32,
}

impl VideoProcessor {
    pub fn new(config_file: &str, motion_threshold: f32, max_ray_distance: f32) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = fs::read_to_string(config_file)?;
        let configs: Vec<VideoConfig> = serde_json::from_str(&config_content)?;
        
        Ok(VideoProcessor {
            configs,
            motion_threshold,
            max_ray_distance,
        })
    }

    pub fn process_frame_sequence(
        &self,
        voxel_grid: &mut VoxelGrid,
        frame_start: usize,
        frame_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        
        for config in &self.configs {
            println!("Processing camera: {}", config.camera_id);
            
            let camera_params: CameraParams = config.camera_params.clone().into();
            
            // In a real implementation, you would use a proper video decoding library
            // For this example, we'll assume frames are extracted as image files
            let frame_dir = format!("{}/frames", config.video_path);
            
            let mut previous_frame: Option<Frame> = None;
            
            for frame_idx in frame_start..(frame_start + frame_count) {
                let frame_path = format!("{}/frame_{:06}.jpg", frame_dir, frame_idx);
                
                if !Path::new(&frame_path).exists() {
                    continue;
                }
                
                let img = image::open(&frame_path)?;
                let current_frame = Frame::from_image(&img);
                
                if let Some(prev_frame) = previous_frame.as_ref() {
                    // Detect motion between frames
                    let motion_mask = MotionMask::from_frames(
                        prev_frame,
                        &current_frame,
                        self.motion_threshold,
                    );
                    
                    // Process motion and update voxel grid
                    process_frame_motion(
                        &motion_mask,
                        &current_frame,
                        &camera_params,
                        voxel_grid,
                        self.max_ray_distance,
                    );
                    
                    println!("Processed frame {} for camera {}", frame_idx, config.camera_id);
                }
                
                previous_frame = Some(current_frame);
            }
        }
        
        Ok(())
    }
}

// Example function to create a sample configuration file
pub fn create_sample_config() -> Result<(), Box<dyn std::error::Error>> {
    let sample_configs = vec![
        VideoConfig {
            camera_id: "camera_1".to_string(),
            video_path: "./videos/camera1".to_string(),
            camera_params: CameraConfigParams {
                position: [-2.0, 0.0, 1.5], // 2m left, 1.5m up
                yaw_degrees: 15.0,   // Slightly angled toward center
                pitch_degrees: -10.0, // Slightly downward
                roll_degrees: 0.0,
                fov_degrees: 60.0,
                width: 1920,
                height: 1080,
            },
        },
        VideoConfig {
            camera_id: "camera_2".to_string(),
            video_path: "./videos/camera2".to_string(),
            camera_params: CameraConfigParams {
                position: [2.0, 0.0, 1.5], // 2m right, 1.5m up
                yaw_degrees: -15.0,  // Slightly angled toward center
                pitch_degrees: -10.0, // Slightly downward
                roll_degrees: 0.0,
                fov_degrees: 60.0,
                width: 1920,
                height: 1080,
            },
        },
        VideoConfig {
            camera_id: "camera_3".to_string(),
            video_path: "./videos/camera3".to_string(),
            camera_params: CameraConfigParams {
                position: [0.0, -3.0, 2.0], // 3m back, 2m up
                yaw_degrees: 0.0,
                pitch_degrees: -20.0, // Looking down more
                roll_degrees: 0.0,
                fov_degrees: 80.0, // Wider field of view
                width: 1920,
                height: 1080,
            },
        },
    ];
    
    let config_json = serde_json::to_string_pretty(&sample_configs)?;
    fs::write("camera_config.json", config_json)?;
    
    println!("Created sample configuration file: camera_config.json");
    Ok(())
}

// Performance analysis utilities
pub fn analyze_voxel_distribution(voxel_grid: &VoxelGrid) {
    let total_voxels = voxel_grid.data.len();
    let non_zero_voxels = voxel_grid.data.iter().filter(|&&x| x > 0.0).count();
    let max_value = voxel_grid.data.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_value = voxel_grid.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let avg_value: f32 = voxel_grid.data.iter().sum::<f32>() / total_voxels as f32;
    
    println!("Voxel Grid Analysis:");
    println!("  Total voxels: {}", total_voxels);
    println!("  Non-zero voxels: {} ({:.2}%)", non_zero_voxels, 
             (non_zero_voxels as f32 / total_voxels as f32) * 100.0);
    println!("  Value range: {:.6} to {:.6}", min_value, max_value);
    println!("  Average value: {:.6}", avg_value);
}

pub fn find_peak_locations(voxel_grid: &VoxelGrid, threshold_percentile: f32) -> Vec<(usize, usize, usize, f32)> {
    // Find voxels above a certain percentile
    let mut sorted_values: Vec<f32> = voxel_grid.data.iter().cloned().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let threshold_idx = ((sorted_values.len() as f32) * (threshold_percentile / 100.0)) as usize;
    let threshold = sorted_values.get(threshold_idx).copied().unwrap_or(0.0);
    
    let mut peaks = Vec::new();
    
    for x in 0..voxel_grid.size {
        for y in 0..voxel_grid.size {
            for z in 0..voxel_grid.size {
                let idx = voxel_grid.get_index(x, y, z);
                let value = voxel_grid.data[idx];
                
                if value > threshold {
                    peaks.push((x, y, z, value));
                }
            }
        }
    }
    
    // Sort by value (highest first)
    peaks.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    
    println!("Found {} peak locations above {:.6} threshold", peaks.len(), threshold);
    
    peaks
}

// Example main function for video processing
pub fn process_videos_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample config if it doesn't exist
    if !Path::new("camera_config.json").exists() {
        create_sample_config()?;
        println!("Please set up your video files and update camera_config.json");
        return Ok(());
    }
    
    // Initialize video processor
    let processor = VideoProcessor::new("camera_config.json", 0.05, 20.0)?;
    
    // Set up voxel grid
    let grid_size = 200;
    let voxel_size = 0.05; // 5cm voxels
    let grid_center = Vector3::new(0.0, 2.0, 1.5); // 2m forward, 1.5m up
    
    let mut voxel_grid = VoxelGrid::new(grid_size, voxel_size, grid_center);
    
    println!("Processing video sequences...");
    
    // Process frames (e.g., frames 0-100)
    processor.process_frame_sequence(&mut voxel_grid, 0, 100)?;
    
    // Analyze results
    analyze_voxel_distribution(&voxel_grid);
    let peaks = find_peak_locations(&voxel_grid, 95.0); // Top 5% of voxels
    
    // Print top 10 peak locations
    for (i, (x, y, z, value)) in peaks.iter().take(10).enumerate() {
        let world_pos = Vector3::new(
            grid_center.x - (grid_size as f32 * voxel_size / 2.0) + (*x as f32 + 0.5) * voxel_size,
            grid_center.y - (grid_size as f32 * voxel_size / 2.0) + (*y as f32 + 0.5) * voxel_size,
            grid_center.z - (grid_size as f32 * voxel_size / 2.0) + (*z as f32 + 0.5) * voxel_size,
        );
        
        println!("Peak {}: Voxel({}, {}, {}) = {:.6}, World({:.3}, {:.3}, {:.3})", 
                 i + 1, x, y, z, value, world_pos.x, world_pos.y, world_pos.z);
    }
    
    // Save results
    voxel_grid.save_to_file("tracking_result.bin")?;
    println!("Saved tracking results to tracking_result.bin");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoxelGrid;

    #[test]
    fn test_video_config_parsing() {
        let config = CameraConfigParams {
            position: [1.0, 2.0, 3.0],
            yaw_degrees: 45.0,
            pitch_degrees: -10.0,
            roll_degrees: 0.0,
            fov_degrees: 60.0,
            width: 1920,
            height: 1080,
        };
        
        let camera_params: CameraParams = config.into();
        assert_eq!(camera_params.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(camera_params.width, 1920);
        assert_eq!(camera_params.height, 1080);
    }
    
    #[test]
    fn test_peak_detection() {
        let mut grid = VoxelGrid::new(10, 1.0, Vector3::new(0.0, 0.0, 0.0));
        
        // Add some test peaks
        grid.add_value(5, 5, 5, 10.0);
        grid.add_value(3, 3, 3, 5.0);
        grid.add_value(7, 7, 7, 15.0);
        
        let peaks = find_peak_locations(&grid, 90.0);
        assert!(peaks.len() > 0);
        
        // Highest peak should be first
        assert_eq!(peaks[0].3, 15.0);
    }
}